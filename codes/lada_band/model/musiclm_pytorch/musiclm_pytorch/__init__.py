try:
    # from musiclm_pytorch.trainer import MuLaNTrainer
    # from musiclm_pytorch.musiclm_pytorch import (
    #     MuLaN,
    #     MuLaNEmbedQuantizer,
    #     MusicLM,
    #     AudioSpectrogramTransformer,
    #     TextTransformer,
    #     SigmoidContrastiveLearning,
    #     SoftmaxContrastiveLearning,
    #     AudioSpectrogramTransformerPretrained,
    #     TextTransformerPretrained,
    #     MuLaNEmbedder,
    # )
    from musiclm_pytorch.musiclm_pytorch import (
        MuLaN,
        AudioSpectrogramTransformer,
        TextTransformer,
        SigmoidContrastiveLearning,
        SoftmaxContrastiveLearning,
        AudioSpectrogramTransformerPretrained,
        TextTransformerPretrained,
        MuLaNEmbedder,
    )
except:
    # from .musiclm_pytorch import (
    #     MuLaN,
    #     MuLaNEmbedQuantizer,
    #     MusicLM,
    #     AudioSpectrogramTransformer,
    #     TextTransformer,
    #     SigmoidContrastiveLearning,
    #     SoftmaxContrastiveLearning,
    #     AudioSpectrogramTransformerPretrained,
    #     TextTransformerPretrained,
    #     MuLaNEmbedder,
    # )
    from .musiclm_pytorch import (
        MuLaN,
        AudioSpectrogramTransformer,
        TextTransformer,
        SigmoidContrastiveLearning,
        SoftmaxContrastiveLearning,
        AudioSpectrogramTransformerPretrained,
        TextTransformerPretrained,
        MuLaNEmbedder,
    )
