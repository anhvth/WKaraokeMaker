from transformers import AutoProcessor, WhisperTokenizer, AutoModelForSpeechSeq2Seq
import whisper
import torch, torch.nn as nn
from ple.all import *
from kmaker.data import wtokenizer
from fastcore.all import patch
import torch.nn.functional as F
from .bbox_utils import *
def get_whisper(model_name):
    return AutoModelForSpeechSeq2Seq.from_pretrained(f"openai/whisper-{model_name}", 
                                                  cache_dir='pretrained/transformers', use_cache=False, 
                                                  local_files_only=False)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

    
def cal_ctc(logits, labels):
    input_lengths = torch.tensor([1500]*len(logits))

    labels_mask = labels >= 0
    target_lengths = labels_mask.sum(-1)
    flattened_targets = labels.masked_select(labels_mask)
    log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
    with torch.backends.cudnn.flags(enabled=False):
        loss = nn.functional.ctc_loss(
            log_probs,
            flattened_targets,
            input_lengths,
            target_lengths,
            blank=109, #w2vmeta['w2vmodel'].config.pad_token_id
            reduction='mean',
            zero_infinity=False,
        )
    return loss

def forward_w2v(
    lm_head,
    input_values,
    attention_mask=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict= None,
    labels= None,
    ):
    

    logits = lm_head(input_values)
    loss = None
    if labels is not None:
        input_lengths = torch.tensor([1500]*len(logits))

        labels_mask = labels >= 0
        target_lengths = labels_mask.sum(-1)
        flattened_targets = labels.masked_select(labels_mask)
        # DB()
        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        with torch.backends.cudnn.flags(enabled=False):
            loss = nn.functional.ctc_loss(
                log_probs,
                flattened_targets,
                input_lengths,
                target_lengths,
                blank=109, #w2vmeta['w2vmodel'].config.pad_token_id
                reduction='mean',
                zero_infinity=False,
            )
    return dict(
        logits=logits,
        loss=loss,
    )




def modify_whisper(model, sot):
    model.model.ctc_lm_head =  nn.Sequential(
        nn.LayerNorm(512),
        MLP(512, 512, 110, 3),
    )
    if sot:
        
        def get_pos_embed_layer(model):
            pos_layer = nn.Linear(512, 1501, bias=False)
            proj_weight = model.proj_out.weight[wtokenizer.timestamp_begin:]
            with torch.no_grad():
                weights = torch.from_numpy(proj_weight.numpy())
            pos_layer.weight = torch.nn.Parameter(data=weights)
            return nn.Sequential(
                pos_layer,
                MLP(1501, 512, 4, 3),
            )
        model.bbox_embed = get_pos_embed_layer(model).requires_grad_(True)
    else:
        model.bbox_embed = MLP(512, 512, 4, 3)
    # model.requires_grad_(False)
    # model.model.encoder.requires_grad_(True)
    # model.model.decoder.layers[-1].requires_grad_(True)

    
    from transformers.models.whisper.modeling_whisper import shift_tokens_right, CrossEntropyLoss
    
    @patch
    def forward_with_ctc(
        self:type(model.model),
        input_features=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        ehs = encoder_outputs[0]
        # ehs = self.hs_dropout(ehs)
        enc_logits = self.ctc_lm_head(ehs)

        # new_hs = self.expand_110_512(enc_logits).log_softmax(2)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=ehs,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return {
            'dec_last_hidden_state':decoder_outputs.last_hidden_state,
            'enc_logits':enc_logits,
        }
    
    
    @patch
    def forward_both(
        self:type(model), # whisper forward wraper with loss
        input_features=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        ctc_labels=None,
        bbox_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model.forward_with_ctc(
            input_features,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.proj_out(outputs['dec_last_hidden_state'])
        bbox_pred = self.bbox_embed(outputs['dec_last_hidden_state']).sigmoid()
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')

            dec_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1), )
        enc_logits=outputs['enc_logits']
        enc_loss = None
        if ctc_labels is not None:
            enc_loss = cal_ctc(enc_logits, ctc_labels)
        bbox_loss = None
        return dict(
            enc_logits = enc_logits,
            enc_loss = enc_loss,
            dec_logits=lm_logits,
            dec_loss = dec_loss,
            bbox_pred=bbox_pred,            
        )
    return model

def calulcate_bbox_loss(src_boxes, target_boxes, loss_scale=None):#, num_boxes):
    
    src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
    target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
    
    F = torch.nn.functional
    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

    losses = {}
    losses['loss_bbox'] = loss_bbox.mean(1)

    loss_giou = 1 - torch.diag(generalized_box_iou(
        src_boxes_xyxy,
        target_boxes_xyxy), )
    losses['loss_giou'] = loss_giou#.sum() / num_boxes
    return losses


    


