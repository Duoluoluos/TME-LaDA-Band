import random
import typing as tp


class FaultTolerantAudioDatasetMixin:
    def _init_fault_tolerance(self, cfg, mode: str) -> None:
        data_cfg = cfg.data
        self.bad_sample_max_retries = max(0, int(getattr(data_cfg, "bad_sample_max_retries", 8)))
        self.bad_sample_log_limit = max(0, int(getattr(data_cfg, "bad_sample_log_limit", 20)))
        self._fault_tolerance_mode = mode
        self._bad_sample_log_count = 0

    def _safe_getitem(
        self,
        index: int,
        load_fn: tp.Callable[[int], tp.Dict[str, tp.Any]],
    ) -> tp.Dict[str, tp.Any]:
        if len(self.songid_list) == 0:
            raise IndexError(f"{self.__class__.__name__} is empty")

        last_exc: tp.Optional[Exception] = None
        tried_indices: tp.Set[int] = set()
        total_attempts = self.bad_sample_max_retries + 1

        for attempt in range(total_attempts):
            candidate_index = index if attempt == 0 else self._sample_retry_index(tried_indices)
            tried_indices.add(candidate_index)
            songid = self.songid_list[candidate_index]

            try:
                item = load_fn(candidate_index)
            except Exception as exc:
                last_exc = exc
                self._log_bad_sample(candidate_index, songid, attempt + 1, total_attempts, exc)
                continue

            if item is None:
                last_exc = RuntimeError(f"loader returned None for songid={songid}")
                self._log_bad_sample(candidate_index, songid, attempt + 1, total_attempts, last_exc)
                continue

            return item

        raise RuntimeError(
            f"failed to fetch a valid sample after {total_attempts} attempts "
            f"(start_index={index})"
        ) from last_exc

    def _sample_retry_index(self, tried_indices: tp.Set[int]) -> int:
        if len(self.songid_list) <= 1:
            return 0

        for _ in range(min(len(self.songid_list), 32)):
            candidate = random.randrange(len(self.songid_list))
            if candidate not in tried_indices:
                return candidate

        remaining = [idx for idx in range(len(self.songid_list)) if idx not in tried_indices]
        if remaining:
            return random.choice(remaining)
        return random.randrange(len(self.songid_list))

    def _log_bad_sample(
        self,
        index: int,
        songid: str,
        attempt: int,
        total_attempts: int,
        exc: Exception,
    ) -> None:
        if self.bad_sample_log_limit == 0:
            return

        if self._bad_sample_log_count < self.bad_sample_log_limit:
            print(
                f"[{self.__class__.__name__}:{self._fault_tolerance_mode}] "
                f"skip bad sample songid={songid} index={index} "
                f"attempt={attempt}/{total_attempts} "
                f"error={self._summarize_exception(exc)}",
                flush=True,
            )
        elif self._bad_sample_log_count == self.bad_sample_log_limit:
            print(
                f"[{self.__class__.__name__}:{self._fault_tolerance_mode}] "
                f"bad sample log limit reached ({self.bad_sample_log_limit}), "
                "suppressing further logs.",
                flush=True,
            )

        self._bad_sample_log_count += 1

    @staticmethod
    def _summarize_exception(exc: Exception) -> str:
        message = str(exc).replace("\n", " ").strip()
        if not message:
            return type(exc).__name__
        return f"{type(exc).__name__}: {message}"
