from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset
import lightning as L
import importlib

class DataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.data.batch_size
        self.num_workers = cfg.data.num_workers
        self.enable_val = bool(cfg.train.get("enable_val", True))
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str):
        dataset_module = importlib.import_module('lada_band.data.{}'.format(self.cfg.data.name))
        if stage in (None, 'fit'):
            if self.train_dataset is None:
                self.train_dataset = dataset_module.ArrangeDataset(self.cfg, mode='train', debug=self.cfg.get('debug'))
            if self.enable_val and self.val_dataset is None:
                self.val_dataset = dataset_module.ArrangeDataset(self.cfg, mode='val', debug=self.cfg.get('debug'))
        elif stage in ('validate', 'test'):
            if self.enable_val and self.val_dataset is None:
                self.val_dataset = dataset_module.ArrangeDataset(self.cfg, mode='val', debug=self.cfg.get('debug'))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn if hasattr(self.train_dataset, 'collate_fn') else None,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True
        )

    def val_dataloader(self):
        if not self.enable_val or self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn if hasattr(self.val_dataset, 'collate_fn') else None,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn if hasattr(self.test_dataset, 'collate_fn') else None,
            num_workers=1,
            persistent_workers=True,
        )
