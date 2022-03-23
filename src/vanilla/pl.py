
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

class PLWrapper(pl.LightningModule):
    def __init__(self, model, config, parser_args, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.args = parser_args
        self.config = config
        self.model = model

        self.train_acc = torchmetrics.Accuracy(dist_sync_on_step=True)
        self.train_acc_top5 = torchmetrics.Accuracy(top_k=5, dist_sync_on_step=True)

        self.val_acc = torchmetrics.Accuracy(dist_sync_on_step=True)
        self.val_acc_top5 = torchmetrics.Accuracy(top_k=5, dist_sync_on_step=True)

        self.test_acc = torchmetrics.Accuracy(dist_sync_on_step=True)
        self.test_acc_top5 = torchmetrics.Accuracy(top_k=5, dist_sync_on_step=True)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=self.args.train_sgd_init_lr, 
            momentum=self.args.train_sgd_momentum, 
            weight_decay=self.args.train_sgd_weight_decay,
            nesterov=True)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.args.train_sgd_init_lr,
                pct_start=0.1,
                epochs=self.args.ft_no_epoch,
                steps_per_epoch=self.args.steps_per_epoch),
            "interval": "step"
        }
        return [optimizer], [scheduler]

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        # Lightning will handle the gradient clipping
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, 1)

        loss = F.cross_entropy(logits, y)
        return {"loss": loss, "preds": preds, "targets": y, "logits": logits}

    def training_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        logits = batch_parts["logits"]
        targets = batch_parts["targets"]
        losses = batch_parts["loss"]

        loss = losses.mean()
        self.train_acc(preds, targets)
        self.train_acc_top5(logits, targets)
        self.log("train/acc", self.train_acc, on_step=True)
        self.log("train/acc5", self.train_acc_top5, on_step=True)
        self.log("train/loss", loss, on_step=True)

        self.current_metric = self.train_acc.compute().item()

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, 1)
        return {"preds": preds, "targets": y, "logits": logits}

    def validation_step_end(self, batch_parts):
        logits = batch_parts["logits"]
        preds = batch_parts["preds"]
        targets = batch_parts["targets"]

        self.val_acc(preds, targets)
        self.val_acc_top5(logits, targets)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)
        self.current_metric = self.val_acc.compute().item()

        self.log("val/acc5", self.val_acc_top5, on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs):
        val_acc = self.val_acc.compute()

        # Save the metric
        self.log('val/acc', val_acc, prog_bar=False)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, 1)
        return {"preds": preds, "targets": y, "logits": logits}

    def test_step_end(self, batch_parts):
        logits = batch_parts["logits"]
        preds = batch_parts["preds"]
        targets = batch_parts["targets"]

        self.test_acc(preds, targets)
        self.test_acc_top5(logits, targets)
        #self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.current_metric = self.test_acc.compute().item()

        #self.log("test/acc5", self.test_acc_top5, on_step=False, on_epoch=True)

    def test_epoch_end(self, outputs):
        test_acc = self.test_acc.compute()

        # Save the metric
        self.log('test/acc', test_acc, prog_bar=False)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        # discard the version number
        items.pop("v_num", None)
        # discard the loss
        items["acc"] = f"{self.current_metric:.3f}"
        # items["acc"] = self.current_metric
        return items
