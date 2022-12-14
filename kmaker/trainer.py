from ple.all import *

from .segment_utils import *


def calulcate_segment_loss(src_boxes, target_boxes):
    src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
    target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)

    F = torch.nn.functional
    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

    losses = {}
    losses["loss_bbox"] = loss_bbox.mean(1)

    loss_giou = 1 - torch.diag(
        generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy),
    )
    losses["loss_giou"] = loss_giou
    return losses


class CustomModelTrainer(LitModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_optimizers(self):
        """
        Setup optimizer and scheduler
        """
        assert self.create_optimizer_fn is not None
        optimizer = self.create_optimizer_fn(self.parameters())

        scheduler = get_scheduler(optimizer, self.create_lr_scheduler_fn)

        return [optimizer], [scheduler]

    def forward(self, batch):
        """
        Forward pass, caculate loss
        """
        outputs = self.model.forward_both(
            batch["inputs"],
            labels=batch["labels"],
            ctc_labels=batch["w2v_labels"],
        )
        bbox_pred = outputs["bbox_pred"][batch["dec_pos"]]
        segment_losses = calulcate_segment_loss(bbox_pred, batch["bboxes"])
        num_bboxes = batch["loss_scale"].sum()
        loss_l1 = (segment_losses["loss_bbox"] * batch["loss_scale"]).sum() / num_bboxes
        loss_giou = (
            segment_losses["loss_giou"] * batch["loss_scale"]
        ).sum() / num_bboxes

        loss_dec = outputs["dec_loss"].mean()
        loss_ctc = outputs["enc_loss"]

        loss_total = loss_l1 + loss_giou + loss_dec + loss_ctc
        return dict(
            loss_l1=loss_l1,
            loss_giou=loss_giou,
            loss_dec=loss_dec,
            loss_ctc=loss_ctc,
            loss_total=loss_total,  # Used for backprop
        )

    def training_step(self, batch, idx):
        out = self(batch)
        for k, v in out.items():
            if k.startswith("loss"):
                self.log(
                    f"train/{k}",
                    v,
                    prog_bar=True,
                    on_epoch=True,
                    on_step=True,
                    sync_dist=True,
                )
        return out["loss_total"]

    def validation_step(self, batch, idx):
        out = self(batch)
        for k, v in out.items():
            if "loss" in k:
                self.log(
                    f"val/{k}",
                    v,
                    prog_bar=True,
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                )
