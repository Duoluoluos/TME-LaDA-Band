import hashlib
import os
import random
import sys
_CODES_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CODES_ROOT)
sys.path.insert(0, os.path.join(_CODES_ROOT, 'MuCodec'))
import argparse
import logging
import shutil
import subprocess
import tempfile
from datetime import datetime
import torch
import omegaconf
from pathlib import Path
import pickle
import numpy as np
import soundfile as sf
from typing import List, Dict, Optional, Tuple
import json
# 导入原始infer.py中的功能
from lada_band.utils.audio_utils import _av_read, convert_audio, AudioFileInfo
from lada_band.utils.config_utils import resolve_config_path, resolve_relative_paths
from lada_band.module import llada as mlm
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# 全局模型列表，用于缓存已加载的模型
model_module_list = [None for _ in range(8)]
DEFAULT_INFER_CONFIG = 'lada_band/conf/infer_1B.yaml'
SUPPORTED_SAVE_FORMATS = ('wav', 'mp3')


def get_segment_duration(start_s: float, end_s: float) -> float:
    """将 [start_s, end_s] 解析成读取 duration；end_s < 0 表示读到文件结束。"""
    if end_s < 0:
        return -1
    duration = end_s - start_s
    if duration <= 0:
        raise ValueError(f"无效的 seg_time: start_s={start_s}, end_s={end_s}，要求 end_s > start_s 或 end_s = -1")
    return duration


def infer_config_from_model_fp(model_fp: str, max_parent_depth: int = 3) -> Optional[str]:
    """尝试从 checkpoint 附近自动推断 config.yaml。"""
    model_path = Path(model_fp).expanduser()
    search_dirs = list(model_path.parents[:max_parent_depth + 1])
    for search_dir in search_dirs:
        for config_name in ('config.yaml', 'config.yml'):
            candidate = search_dir / config_name
            if candidate.exists():
                return str(candidate)
    return None


def resolve_infer_config_path(config_arg: Optional[str], model_fp: str) -> str:
    """优先使用显式配置；未提供时尝试从 model_fp 邻近目录自动推断。"""
    if config_arg:
        config_fp = resolve_config_path(DEFAULT_INFER_CONFIG, config_arg)
        config_path = Path(config_fp).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_arg}")
        return str(config_path)

    inferred_config_fp = infer_config_from_model_fp(model_fp)
    if inferred_config_fp:
        logger.info(f"未显式传入配置文件，自动使用: {inferred_config_fp}")
        return inferred_config_fp

    raise FileNotFoundError(
        "未找到可用配置文件。请显式传 `--config /abs/path/to/config.yaml`，"
        "或把 checkpoint 放在含有 config.yaml 的实验目录下。"
    )


def extract_legacy_config_override(extra_args: List[str]) -> Tuple[Optional[str], List[str]]:
    """兼容 config_fp=... / +config_fp=... / cfg_fn=... 的旧写法。"""
    config_override = None
    remaining_args = []
    for arg in extra_args:
        if '=' not in arg:
            remaining_args.append(arg)
            continue

        key, value = arg.split('=', 1)
        normalized_key = key[1:] if key.startswith('+') else key
        if normalized_key in {'config', 'config_fp', 'cfg_fn'}:
            config_override = value
            continue

        remaining_args.append(arg)

    return config_override, remaining_args


def expand_cli_values(values: Optional[List[str]], expected_len: int, arg_name: str, default_value=None) -> List[Optional[str]]:
    """将单值广播到所有样本，或校验一一对齐的多值输入。"""
    if values is None:
        return [default_value] * expected_len
    if len(values) == 1 and expected_len > 1:
        return values * expected_len
    if len(values) != expected_len:
        raise ValueError(
            f"{arg_name} 的数量需要是 1 或 {expected_len}，"
            f"当前收到 {len(values)} 个。"
        )
    return values


def get_condition_type(ref_audio: Optional[str], text: Optional[str]) -> str:
    if ref_audio:
        return 'audio'
    if text:
        return 'text'
    return 'none'


def build_direct_test_data(args) -> List[Dict]:
    """直接从 vocal/text/ref 参数构建推理样本，不依赖 meta/test_fp。"""
    text_list = expand_cli_values(args.texts, len(args.vocal_fps), '--texts', default_value='')
    ref_audio_list = expand_cli_values(args.ref_audios, len(args.vocal_fps), '--ref_audios', default_value=None)

    test_data = []
    for idx, voc_fp in enumerate(args.vocal_fps):
        voc_path = str(Path(voc_fp).expanduser())
        ref_audio = ref_audio_list[idx]
        if ref_audio is not None:
            ref_audio = str(Path(ref_audio).expanduser())

        test_data.append({
            'songid': Path(voc_path).stem,
            'voc_fp': voc_path,
            'text': text_list[idx] or '',
            'duration': 0,
            'ref_audio': ref_audio,
            'ref_songid': Path(ref_audio).stem if ref_audio else None,
            'sample_idx': idx,
        })

    logger.info(f"使用直接输入模式，共 {len(test_data)} 条 vocal")
    return test_data


def build_test_data(args, cfg: omegaconf.DictConfig) -> List[Dict]:
    if args.vocal_fps:
        if args.ugc_dir:
            logger.warning("直接输入模式下会忽略 --ugc_dir")
        return build_direct_test_data(args)

    if args.ugc_dir:
        logger.info(f"使用UGC目录覆盖原voc_dir: {args.ugc_dir}")
        cfg.data.voc_dir = args.ugc_dir

    return load_test_data(cfg, specific_songids=args.songids)


def resolve_reference_audio(voc_fp: str, data: Dict, cfg: omegaconf.DictConfig, args) -> Tuple[Optional[str], Optional[str]]:
    """按 样本级 > 全局 > 数据集自动匹配 的优先级解析参考音频。"""
    if args.ablation_no_ref:
        return None, None

    sample_ref_audio = data.get('ref_audio')
    sample_ref_songid = data.get('ref_songid')
    if sample_ref_audio:
        if os.path.exists(sample_ref_audio):
            return sample_ref_audio, sample_ref_songid or Path(sample_ref_audio).stem
        logger.warning(f"样本级参考音频不存在: {sample_ref_audio}，将跳过")

    if args.global_ref:
        if os.path.exists(args.global_ref):
            return args.global_ref, Path(args.global_ref).stem
        logger.warning(f"全局参考音频不存在: {args.global_ref}，将跳过")

    if args.vocal_fps:
        return None, None

    ref_audio, ref_songid = get_corresponding_acc(voc_fp, cfg)
    if ref_audio and not os.path.exists(ref_audio):
        logger.warning(f"对应伴奏文件不存在: {ref_audio}，跳过使用参考音频")
        return None, None
    return ref_audio, ref_songid


def resolve_text_input(data: Dict, args) -> str:
    """解析最终参与生成的文本条件。"""
    if args.ablation_no_text:
        return ''
    if args.global_text is not None:
        return args.global_text
    return data.get('text', '') or ''


def derive_sample_seed(seed: int, sequence_id: str) -> int:
    """Derive a stable per-sample seed from the global seed and song id."""
    digest = hashlib.blake2b(f"{int(seed)}|{sequence_id}".encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'big') % (2**31 - 1)


def set_generation_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit(repo_dir: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def audio_duration_seconds(audio_fp: Optional[str]) -> Optional[float]:
    if not audio_fp:
        return None
    try:
        return float(sf.info(audio_fp).duration)
    except Exception as exc:
        logger.warning(f"读取音频时长失败 {audio_fp}: {exc}")
        return None


def init_model(model_fp: str, config_fp: str, model_num: int = 0) -> mlm.ModelModule:
    """初始化模型并缓存"""

    model_num = 0

    global model_module_list
    if model_module_list[model_num] is None:
        try:
            cfg_raw = omegaconf.OmegaConf.load(config_fp)
            cfg_dict = omegaconf.OmegaConf.to_container(cfg_raw, resolve=True)
            cfg_dict = resolve_relative_paths(cfg_dict, config_file_path=config_fp)
            cfg = omegaconf.OmegaConf.create(cfg_dict)
            model_module = mlm.ModelModule(cfg)
            model_module.load_model(model_fp)

            # 强制放到 cuda:0
            model_module = model_module.cuda(model_num)

            model_module.eval()
            model_module_list[model_num] = model_module
            logger.info(f"成功加载模型 {model_fp} 到 GPU {model_num} (物理卡由环境变量控制)")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    return model_module_list[model_num]



def load_test_data(cfg: omegaconf.DictConfig, specific_songids: List[str] = None) -> List[Dict]:
    """加载测试集数据，返回包含vocal文件路径和参考文本的列表"""
    try:
        # 读取元数据
        with open(cfg.data.meta_fp, 'rb') as f:
            metas: Dict[str, AudioFileInfo] = pickle.load(f)

        # 读取测试集songid列表
        if specific_songids:
            songid_list = specific_songids
            logger.info(f"使用指定的 {len(songid_list)} 个songid进行测试")
        else:
            songid_fp = cfg.data.test_fp if hasattr(cfg.data, 'test_fp') else cfg.data.val_fp
            with open(songid_fp, 'r') as f:
                songid_list = f.read().splitlines()

        # 构建测试数据列表
        test_data = []
        have_subdir = getattr(cfg.data, 'have_subdir', False)

        for songid in songid_list:
            if songid in metas:
                meta = metas[songid]
                # 构建vocal文件路径
                if have_subdir and hasattr(meta, 'song_relative_dir'):
                    voc_fp = os.path.join(
                        cfg.data.voc_dir,
                        meta.song_relative_dir,
                        f"{songid}.{cfg.data.voc_suffix}"
                    )
                else:
                    voc_fp = os.path.join(cfg.data.voc_dir, f"{songid}.{cfg.data.voc_suffix}")

                test_data.append({
                    'songid': songid,
                    'voc_fp': voc_fp,
                    'text': getattr(meta, 'text', ''),
                    'duration': getattr(meta, 'duration', 0)
                })

        logger.info(f"成功加载 {len(test_data)} 个测试样本")
        return test_data
    except Exception as e:
        logger.error(f"加载测试数据失败: {str(e)}")
        raise

def get_corresponding_acc(voc_fp: str, cfg: omegaconf.DictConfig) -> Tuple[str, str]:
    """获取与人声文件对应的伴奏文件

    参数:
    voc_fp: 人声文件路径
    cfg: 配置对象

    返回:
    (伴奏文件路径, songid)
    """
    try:
        # 从人声文件路径中提取songid
        songid = os.path.basename(voc_fp).split('.')[0]

        # 根据配置构建伴奏文件路径
        if hasattr(cfg.data, 'have_subdir') and cfg.data.have_subdir and hasattr(cfg.data, 'acc_dir'):
            # 如果配置中提到有子目录，尝试从元数据中获取目录信息
            # 这里我们假设结构与voc_dir相同
            voc_dir = cfg.data.voc_dir
            if voc_dir in voc_fp:
                # 替换voc_dir为acc_dir
                acc_dir = cfg.data.acc_dir
                acc_fp = voc_fp.replace(voc_dir, acc_dir)
                # 替换文件后缀
                voc_suffix = cfg.data.voc_suffix if hasattr(cfg.data, 'voc_suffix') else 'wav'
                acc_suffix = cfg.data.acc_suffix if hasattr(cfg.data, 'acc_suffix') else 'wav'
                acc_fp = acc_fp.replace(f'.{voc_suffix}', f'.{acc_suffix}')
                return acc_fp, songid

        # 默认方式构建伴奏路径
        acc_dir = getattr(cfg.data, 'acc_dir', None)
        if acc_dir:
            acc_suffix = getattr(cfg.data, 'acc_suffix', 'wav')
            acc_fp = os.path.join(acc_dir, f"{songid}.{acc_suffix}")
            return acc_fp, songid

        logger.warning(f"无法找到与人声 {voc_fp} 对应的伴奏文件，配置中缺少acc_dir")
        return None, None
    except Exception as e:
        logger.error(f"获取对应伴奏文件失败: {str(e)}")
        return None, None

def generate(input_fp: str, start_s: float, end_s: float, model_module: mlm.ModelModule,
                   ref_audio: str = None, text: str = None, ref_audio_duration: float = -1,
                   decode_mode: str = 'diffusion', **kwargs) -> torch.Tensor:
    """生成音频"""
    segment_duration = get_segment_duration(start_s, end_s)
    voc_audio, sr = _av_read(input_fp, seek_time=start_s, duration=segment_duration)
    voc_audio = convert_audio(voc_audio, from_rate=sr, to_rate=48000, to_channels=2)

    # 添加噪声
    # if hasattr(model_module.cfg.data, 'noise') and model_module.cfg.data.noise.name == 'gaussian':
    #     noise = torch.randn_like(voc_audio) * model_module.cfg.data.noise.sigma
    #     voc_audio = voc_audio + noise

    # voc_audio = voc_audio.to(model_module.device)

    # 处理参考音频
    ref_audio_tensor = None
    if ref_audio:
        ref_audio_tensor, sr = _av_read(ref_audio, seek_time=start_s, duration=ref_audio_duration)
        ref_audio_tensor = convert_audio(ref_audio_tensor, from_rate=sr, to_rate=48000, to_channels=1)
    # 生成音频
    audio = model_module.generate_with_clamp_and_codec(
        voc_audio=voc_audio,
        text=text,
        ref_audio=ref_audio_tensor,
        decode_mode=decode_mode,
        **kwargs
    )

    return audio


def normalize_save_format(audio_format: str) -> str:
    """规范化音频保存格式。"""
    normalized = (audio_format or 'mp3').lower().lstrip('.')
    if normalized not in SUPPORTED_SAVE_FORMATS:
        raise ValueError(f"不支持的保存格式: {audio_format}，当前仅支持: {', '.join(SUPPORTED_SAVE_FORMATS)}")
    return normalized


def write_audio_file(audio, samplerate: int, save_fp: Path, audio_format: str) -> str:
    """写出音频文件；mp3 通过 ffmpeg 转码，wav 直接写。"""
    audio_format = normalize_save_format(audio_format)
    save_fp = save_fp.with_suffix(f'.{audio_format}')

    if audio_format == 'wav':
        sf.write(str(save_fp), data=audio, samplerate=samplerate)
        return str(save_fp)

    ffmpeg_bin = shutil.which('ffmpeg')
    if ffmpeg_bin is None:
        raise RuntimeError("保存 mp3 需要 ffmpeg，但当前环境未找到 ffmpeg。")

    tmp_wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name

        sf.write(tmp_wav_path, data=audio, samplerate=samplerate)
        cmd = [
            ffmpeg_bin,
            '-y',
            '-i',
            tmp_wav_path,
            '-c:a',
            'libmp3lame',
            '-q:a',
            '2',
            '-loglevel',
            'error',
            str(save_fp),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return str(save_fp)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode('utf-8', errors='ignore').strip() if exc.stderr else str(exc)
        raise RuntimeError(f"ffmpeg 转码 mp3 失败: {stderr}") from exc
    finally:
        if tmp_wav_path and os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)


def save_audio(input_fp: str, model_fp: str, save_dir: str, audio: torch.Tensor,
               start_s: float, ref_audio: str, text: str, ref_songid: str = None,
               audio_format: str = 'mp3') -> str:
    """保存生成的音频文件"""
    try:
        import time

        # 创建输出目录
        os.makedirs(save_dir, exist_ok=True)

        # 提取信息构建文件名
        songid = os.path.basename(input_fp).split('.')[0]
        model_sig = os.path.basename(model_fp).split('-')[0]
        cond_type = get_condition_type(ref_audio, text)
        cond_id = ref_songid or (text[:20].replace(' ', '_') if text else 'none')
        timestamp = str(int(time.time()))[-3:]
        audio_format = normalize_save_format(audio_format)

        save_fn = f'model_{model_sig}_voc={songid}_start={int(start_s)}_cond_{cond_type}={cond_id}_tm_{timestamp}.{audio_format}'
        save_fp = Path(save_dir) / save_fn

        # 保存音频
        if isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy().T
        else:
            audio_np = audio

        save_fp = write_audio_file(audio=audio_np, samplerate=48000, save_fp=save_fp, audio_format=audio_format)

        logger.info(f"已保存音频: {save_fp}")
        return save_fp
    except Exception as e:
        logger.error(f"保存音频失败: {str(e)}")
        raise


def add_vocal_to_generated(
    generated_audio,
    vocal_fp: str,
    start_s: float,
    end_s: float,
    target_sr: int = 48000,
):
    """将生成音频与对应 vocal 片段直接相加，不做额外混音处理。"""
    if isinstance(generated_audio, torch.Tensor):
        generated_audio = generated_audio.detach().cpu().numpy()

    generated_audio = np.asarray(generated_audio)
    generated_audio = np.squeeze(generated_audio)

    if generated_audio.ndim == 1:
        generated_audio = generated_audio[:, np.newaxis]
    elif generated_audio.ndim == 2 and generated_audio.shape[0] < generated_audio.shape[1]:
        generated_audio = generated_audio.transpose()
    elif generated_audio.ndim != 2:
        raise ValueError(f"不支持的 generated_audio 维度: {generated_audio.shape}")

    segment_duration = get_segment_duration(start_s, end_s)
    vocal_audio, vocal_sr = _av_read(vocal_fp, seek_time=start_s, duration=segment_duration)
    if vocal_audio.shape[-1] == 0:
        raise RuntimeError("vocals音频长度为0")

    vocal_audio = convert_audio(
        vocal_audio,
        from_rate=vocal_sr,
        to_rate=target_sr,
        to_channels=1,
    ).numpy().T

    if vocal_audio.ndim == 1:
        vocal_audio = vocal_audio[:, np.newaxis]

    if generated_audio.shape[1] != vocal_audio.shape[1]:
        if vocal_audio.shape[1] == 1:
            vocal_audio = np.repeat(vocal_audio, generated_audio.shape[1], axis=1)
        elif generated_audio.shape[1] == 1:
            generated_audio = np.repeat(generated_audio, vocal_audio.shape[1], axis=1)
        else:
            raise ValueError(
                f"声道数不匹配: generated={generated_audio.shape[1]}, vocal={vocal_audio.shape[1]}"
            )

    min_len = min(len(generated_audio), len(vocal_audio))
    return generated_audio[:min_len] + vocal_audio[:min_len]


def save_audio_segment(input_fp: str, start_s: float, end_s: float, save_fp: Path, audio_format: str = 'mp3') -> str:
    """按给定时间范围裁切音频片段并保存。"""
    try:
        segment_duration = get_segment_duration(start_s, end_s)
        audio, sr = _av_read(input_fp, seek_time=start_s, duration=segment_duration)
        audio = audio.numpy().T

        save_fp.parent.mkdir(parents=True, exist_ok=True)
        audio_format = normalize_save_format(audio_format)
        save_fp = Path(write_audio_file(audio=audio, samplerate=sr, save_fp=save_fp, audio_format=audio_format))
        logger.info(f"已保存音频片段: {save_fp}")
        return str(save_fp)
    except Exception as e:
        logger.error(f"保存音频片段失败 {input_fp}: {str(e)}")
        raise

def batch_generate(args):
    """批量生成音频的主函数"""
    if args.save_format == 'mp3' and shutil.which('ffmpeg') is None:
        raise RuntimeError("当前环境未找到 ffmpeg，无法按默认 mp3 格式保存。可安装 ffmpeg，或改用 --save_format wav。")

    # 加载配置
    cfg_raw = omegaconf.OmegaConf.load(args.config_fp)
    cfg_dict = omegaconf.OmegaConf.to_container(cfg_raw, resolve=True)
    cfg_dict = resolve_relative_paths(cfg_dict, config_file_path=args.config_fp)
    cfg = omegaconf.OmegaConf.create(cfg_dict)
    cfg_infer = cfg.get('infer', None) if cfg is not None else None
    default_decode_mode = 'diffusion'
    if cfg_infer is not None:
        default_decode_mode = cfg_infer.get('decode_mode', default_decode_mode)
    resolved_decode_mode = args.decode_mode or default_decode_mode

    # 加载测试数据
    test_data = build_test_data(args, cfg)

    # 解析时间参数
    start_s, end_s = args.seg_time
    selected_test_data = test_data[:args.max_samples]

    # 创建输出目录结构
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')

    # 初始化模型
    model_module = init_model(args.model_fp, args.config_fp)
    # 保存生成伴奏结果；默认与对应 vocal 相加，--acc_only 时直接保存伴奏
    generated_dir = output_base_dir / "generated_acc"
    generated_dir.mkdir(exist_ok=True)
    vocal_segment_dir = output_base_dir / "vocal_segments"
    reference_segment_dir = output_base_dir / "reference_segments"
    segment_end_tag = 'end' if end_s < 0 else int(end_s)

    if args.acc_only:
        vocal_segment_dir.mkdir(exist_ok=True)
        reference_segment_dir.mkdir(exist_ok=True)

    # 创建结果记录文件
    results_record = []
    results_file = output_base_dir / f"generation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # 记录是否进行消融实验（不加参考音频）
    if args.ablation_no_ref:
        logger.info("进行消融实验：强制不使用任何参考音频")
    # 记录是否进行消融实验（不加文本输入）
    if args.ablation_no_text:
        logger.info("进行消融实验：强制不使用任何文本输入")

    gen_kwargs = {
        'top_k': args.top_k,
        'top_p': args.top_p,
        'sampling_steps': args.sampling_steps,
        'temperature': args.temperature,
        'mask_temperature': args.mask_temperature,
        'remask_schedule': args.remask_schedule,
        'remask_strategy': args.remask_strategy,
        'remask_seed': args.seed,
        'decode_mode': resolved_decode_mode,
    }

    run_config = {
        'model_fp': args.model_fp,
        'config_fp': args.config_fp,
        'output_dir': str(output_base_dir),
        'commit_hash': get_git_commit(Path(__file__).resolve().parent),
        'checkpoint': args.model_fp,
        'strategy': args.remask_strategy,
        'sampling_steps': args.sampling_steps,
        'schedule': args.remask_schedule,
        'seed': args.seed,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'mask_temperature': args.mask_temperature,
        'decode_mode': resolved_decode_mode,
        'ablation_no_ref': args.ablation_no_ref,
        'ablation_no_text': args.ablation_no_text,
        'seg_time': [start_s, end_s],
        'sample_ids': [item.get('songid') for item in selected_test_data],
    }
    run_config_file = output_base_dir / 'run_config.json'
    with open(run_config_file, 'w', encoding='utf-8') as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    logger.info(f"运行配置保存在: {run_config_file}")

    remask_trace_file = output_base_dir / 'remask_trace.jsonl'
    if remask_trace_file.exists():
        remask_trace_file.unlink()
    # 处理每个测试样本
    total_samples = len(selected_test_data)
    for idx, data in enumerate(selected_test_data):
        voc_fp = data['voc_fp']
        text = resolve_text_input(data, args)
        songid = data['songid']

        logger.info(f"处理样本 {idx+1}/{total_samples}: {songid}")

        ref_audio, ref_songid = None, None
        generated_fp = None
        saved_vocal_fp = None
        saved_ref_fp = None
        elapsed_time = None

        try:
            if not os.path.exists(voc_fp):
                raise FileNotFoundError(f"文件不存在: {voc_fp}")

            ref_audio, ref_songid = resolve_reference_audio(voc_fp, data, cfg, args)
            if ref_audio:
                logger.info(f"使用参考音频: {ref_audio}")
            elif not args.vocal_fps:
                logger.warning("未找到对应伴奏，跳过使用参考音频")

            if text:
                logger.info(f"使用文本输入: {text}")

            if args.acc_only:
                saved_vocal_fp = save_audio_segment(
                    input_fp=voc_fp,
                    start_s=start_s,
                    end_s=end_s,
                    save_fp=vocal_segment_dir / f"{songid}_start={int(start_s)}_end={segment_end_tag}.{args.save_format}",
                    audio_format=args.save_format,
                )
                if ref_audio:
                    saved_ref_fp = save_audio_segment(
                        input_fp=ref_audio,
                        start_s=start_s,
                        end_s=end_s,
                        save_fp=reference_segment_dir / f"{ref_songid}_start={int(start_s)}_end={segment_end_tag}.{args.save_format}",
                        audio_format=args.save_format,
                    )

            import time
            sample_seed = derive_sample_seed(args.seed, songid)
            set_generation_seed(sample_seed)
            sample_remask_trace = []
            sample_gen_kwargs = dict(gen_kwargs, remask_sequence_id=songid)
            start_time = time.time()

            with torch.inference_mode():
                generated_audio = generate(
                    input_fp=voc_fp,
                    start_s=start_s,
                    end_s=end_s,
                    model_module=model_module,
                    ref_audio=ref_audio,
                    text=text,
                    ref_audio_duration=args.ref_audio_duration,
                    remask_trace=sample_remask_trace,
                    **sample_gen_kwargs
                )

                elapsed_time = time.time() - start_time
                logger.info(f"生成耗时: {elapsed_time:.2f} 秒")

                if hasattr(generated_audio, 'cpu'):
                    generated_audio = generated_audio.detach().cpu().numpy()

                generated_audio = generated_audio.squeeze()

                if generated_audio.ndim == 2 and generated_audio.shape[0] < generated_audio.shape[1]:
                    generated_audio = generated_audio.transpose()

                if generated_audio.ndim == 3:
                    generated_audio = generated_audio.squeeze()

                if not args.acc_only:
                    generated_audio = add_vocal_to_generated(
                        generated_audio=generated_audio,
                        vocal_fp=voc_fp,
                        start_s=start_s,
                        end_s=end_s,
                    )

                generated_fp = save_audio(
                    input_fp=voc_fp,
                    model_fp=args.model_fp,
                    save_dir=str(generated_dir),
                    audio=generated_audio,
                    start_s=start_s,
                    ref_audio=ref_audio,
                    text=text,
                    ref_songid=ref_songid,
                    audio_format=args.save_format,
                )

            duration_seconds = audio_duration_seconds(generated_fp)
            rtf = elapsed_time / duration_seconds if elapsed_time and duration_seconds and duration_seconds > 0 else None

            with open(remask_trace_file, 'a', encoding='utf-8') as f:
                for trace_entry in sample_remask_trace:
                    f.write(json.dumps(trace_entry, ensure_ascii=False) + '\n')

            result_entry = {
                'voc_songid': songid,
                'voc_fp': voc_fp,
                'reference_type': get_condition_type(ref_audio, text),
                'reference_songid': ref_songid,
                'reference_fp': ref_audio,
                'text': text,
                'generated_fp': generated_fp,
                'saved_vocal_fp': saved_vocal_fp,
                'saved_ref_fp': saved_ref_fp,
                'acc_only': args.acc_only,
                'start_time': start_s,
                'end_time': end_s,
                'generation_time_seconds': elapsed_time,
                'audio_duration_seconds': duration_seconds,
                'rtf': rtf,
                'sample_seed': sample_seed,
                'generation_params': sample_gen_kwargs,
                'status': 'success',
                'error': None,
            }
        except Exception as exc:
            logger.exception(f"推理样本失败，已跳过: {songid}")
            result_entry = {
                'voc_songid': songid,
                'voc_fp': voc_fp,
                'reference_type': get_condition_type(ref_audio, text),
                'reference_songid': ref_songid,
                'reference_fp': ref_audio,
                'text': text,
                'generated_fp': generated_fp,
                'saved_vocal_fp': saved_vocal_fp,
                'saved_ref_fp': saved_ref_fp,
                'acc_only': args.acc_only,
                'start_time': start_s,
                'end_time': end_s,
                'generation_time_seconds': elapsed_time,
                'audio_duration_seconds': None,
                'rtf': None,
                'sample_seed': derive_sample_seed(args.seed, songid),
                'generation_params': dict(gen_kwargs, remask_sequence_id=songid),
                'status': 'failed',
                'error': str(exc),
            }
        finally:
            results_record.append(result_entry)
            if (idx + 1) % 10 == 0:
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results_record, f, ensure_ascii=False, indent=2)
                logger.info(f"已保存中间结果到 {results_file}")
            torch.cuda.empty_cache()

    # 最终保存所有结果
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_record, f, ensure_ascii=False, indent=2)
    logger.info(f"批量生成完成！")
    if args.acc_only:
        logger.info(f"生成伴奏结果保存在: {generated_dir}")
        logger.info(f"Vocal 片段保存在: {vocal_segment_dir}")
        logger.info(f"参考音频片段保存在: {reference_segment_dir}")
    else:
        logger.info(f"生成伴奏与对应vocal直接相加后的结果保存在: {generated_dir}")
    logger.info(f"结果记录保存在: {results_file}")


def parse_infer_args():
    parser = argparse.ArgumentParser(
        description='批量音频生成脚本',
        epilog=(
            "Examples:\n"
            "  python infer.py --model_fp /abs/exp/hash/ckpt-step=100.ckpt --config lada_band/conf/config_dj.yaml \\\n"
            "    --output_dir ./demo/direct --seg_time 0 30 --vocal_fps /abs/a.wav /abs/b.wav \\\n"
            "    --texts \"energetic edm\" \"warm city-pop\"\n"
            "\n"
            "  # AR ablation demo\n"
            "  python infer.py --model_fp /abs/exp/hash/ckpt-step=100.ckpt --config /abs/ar_config.yaml --output_dir ./demo/ar \\\n"
            "    --seg_time 0 30 --vocal_fps /abs/a.wav --texts \"bright dance pop\"\n"
            "\n"
            "  # 如果 checkpoint 所在实验目录下已有 config.yaml，可省略 --config\n"
            "  python infer.py --model_fp /abs/exp/hash/ckpt-step=100.ckpt --output_dir ./demo/direct \\\n"
            "    --seg_time 0 30 --vocal_fps /abs/a.wav --texts \"bright dance pop\""
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--model_fp', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--config', type=str, help='推荐入口：配置文件路径，支持绝对路径、相对路径或 lada_band/conf 下的文件名')
    parser.add_argument('--config_fp', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--seg_time', nargs=2, type=float, metavar=('START_S', 'END_S'), required=True, help='音频片段时间 [start_s end_s]；end_s=-1 表示读到文件结束')
    parser.add_argument('--ref_audio_duration', type=float, default=-1, help='参考音频读取时长，默认为-1（全部读取）')
    parser.add_argument('--top_k', type=int, default=100, help='top_k采样参数')
    parser.add_argument('--top_p', type=float, default=0.9, help='top_p采样参数')
    parser.add_argument('--sampling_steps', type=int, default=60, help='采样步数')
    parser.add_argument('--num_steps', type=int, default=None, help='--sampling_steps 的别名；提供时覆盖 --sampling_steps')
    parser.add_argument('--remask_strategy', type=str, default='low_confidence', choices=['low_confidence', 'random', 'fixed_position', 'one_shot', 'no_remask'], help='remasking 选择策略')
    parser.add_argument('--remask_schedule', '--schedule', dest='remask_schedule', type=str, default='cosine', help='remask schedule，如 cosine/linear/sqrt/square/cubic/power:<float>')
    parser.add_argument('--seed', type=int, default=13, help='全局随机种子；每个样本会由 seed 和 songid 派生独立 seed')
    parser.add_argument('--temperature', type=float, default=1.0, help='温度参数')
    parser.add_argument('--mask_temperature', type=float, default=10.5, help='掩码温度参数')
    parser.add_argument('--decode_mode', type=str, default=None, help='解码方式: diffusion 或 ar/autoregressive；省略时优先读配置里的 infer.decode_mode，否则默认 diffusion')
    parser.add_argument('--save_format', type=normalize_save_format, default='mp3', choices=SUPPORTED_SAVE_FORMATS, help='输出音频保存格式，默认 mp3；支持 wav/mp3')
    # 消融实验参数
    parser.add_argument('--ablation_no_ref', action='store_true', default=False, help='消融实验：强制不使用任何参考音频')
    parser.add_argument('--ablation_no_text', action='store_true', default=False, help='消融实验：强制不使用任何文本输入')
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数，用于调试')
    parser.add_argument('--acc_only', action='store_true', default=False, help='仅保存生成的伴奏，并额外导出按 seg_time 裁切后的 vocal 与 ref_audio 片段')
    # 数据集模式参数
    parser.add_argument('--songids', nargs='+', help='指定要处理的songid列表，如果指定则忽略测试集文件')
    parser.add_argument('--ugc_dir', type=str, help='指定UGC目录，如果不为空，则覆盖配置文件中的voc_dir')
    # 直接输入模式参数
    parser.add_argument('--vocal_fps', nargs='+', help='直接指定要处理的 vocal 文件路径列表')
    parser.add_argument('--texts', nargs='+', help='与 --vocal_fps 对齐的文本列表；传 1 条时会广播到全部 vocal')
    parser.add_argument('--ref_audios', nargs='+', help='与 --vocal_fps 对齐的参考音频列表；传 1 条时会广播到全部 vocal')
    # 全局覆盖参数
    parser.add_argument('--global_ref', type=str, help='指定一个全局参考音频文件，优先级低于 --ref_audios')
    parser.add_argument('--global_text', type=str, help='指定一个全局文本输入，会覆盖样本级文本')

    args, unknown = parser.parse_known_args()

    config_override, remaining_unknown = extract_legacy_config_override(unknown)
    if remaining_unknown:
        parser.error(f"不支持的额外参数: {' '.join(remaining_unknown)}")

    if args.songids and args.vocal_fps:
        parser.error("--songids 和 --vocal_fps 不能同时使用；前者是数据集模式，后者是直接输入模式。")
    if args.texts and not args.vocal_fps:
        parser.error("--texts 仅在 --vocal_fps 直接输入模式下使用。")
    if args.ref_audios and not args.vocal_fps:
        parser.error("--ref_audios 仅在 --vocal_fps 直接输入模式下使用。")
    if args.num_steps is not None:
        if args.num_steps <= 0:
            parser.error("--num_steps 必须大于 0。")
        args.sampling_steps = args.num_steps
    if args.sampling_steps <= 0:
        parser.error("--sampling_steps 必须大于 0。")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples 必须大于 0。")
    if args.vocal_fps:
        try:
            expand_cli_values(args.texts, len(args.vocal_fps), '--texts', default_value='')
            expand_cli_values(args.ref_audios, len(args.vocal_fps), '--ref_audios', default_value=None)
        except ValueError as exc:
            parser.error(str(exc))

    args.model_fp = str(Path(args.model_fp).expanduser())
    args.output_dir = str(Path(args.output_dir).expanduser())
    if args.global_ref:
        args.global_ref = str(Path(args.global_ref).expanduser())
    if args.ugc_dir:
        args.ugc_dir = str(Path(args.ugc_dir).expanduser())

    try:
        get_segment_duration(*args.seg_time)
    except ValueError as exc:
        parser.error(str(exc))

    config_arg = config_override or args.config or args.config_fp
    try:
        args.config_fp = resolve_infer_config_path(config_arg, args.model_fp)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    return args


if __name__ == '__main__':
    batch_generate(parse_infer_args())
