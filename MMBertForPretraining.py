from re import L
from typing import List, Tuple

import torch
from torch import nn

from transformers import BertForPreTraining
from transformers.modeling_bert import BertPreTrainingHeads, BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler 

from MMBertEmbedding import JointEmbeddings

class MMBertModel(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.config = config
        #self.jointEmbeddings = JointEmbeddings(config.hidden_size,0.5,'mosi')
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def set_joint_embeddings(self, dataset):
        self.dataset = dataset
        self.jointEmbeddings = JointEmbeddings(self.config.hidden_size, 0.5, dataset)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self,value):
        self.embeddings.word_embeddings = value
    
    def _prune_heads(self,heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, torch.Tensor]]:
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    #From huggingface docu
    def get_extended_attention_mask(self, attention_mask, input_shape, device, joint):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if joint:
            if attention_mask.dim() == 3:
                attention_mask = torch.narrow(attention_mask,2,0,1).squeeze(-1)
                extended_attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 2:
                # Provided a padding mask of dimensions [batch_size, seq_length]
                # - if the model is a decoder, apply a causal mask in addition to the padding mask
                # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
                if self.config.is_decoder:
                    batch_size, seq_length = input_shape
                    seq_ids = torch.arange(seq_length, device=device)
                    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                    # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                    # causal and attention masks must have same type with pytorch version < 1.3
                    causal_mask = causal_mask.to(attention_mask.dtype)

                    if causal_mask.shape[1] < attention_mask.shape[1]:
                        prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                        causal_mask = torch.cat(
                            [
                                torch.ones(
                                    (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                                ),
                                causal_mask,
                            ],
                            axis=-1,
                        )

                    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                else:
                    extended_attention_mask = attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "You have so large dimension (), Check dimension or shape ".format(attention_mask.dim())
                    )
        else:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.mean(2)
                extended_attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 2:
                # Provided a padding mask of dimensions [batch_size, seq_length]
                # - if the model is a decoder, apply a causal mask in addition to the padding mask
                # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
                if self.config.is_decoder:
                    batch_size, seq_length = input_shape
                    seq_ids = torch.arange(seq_length, device=device)
                    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                    # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                    # causal and attention masks must have same type with pytorch version < 1.3
                    causal_mask = causal_mask.to(attention_mask.dtype)

                    if causal_mask.shape[1] < attention_mask.shape[1]:
                        prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                        causal_mask = torch.cat(
                            [
                                torch.ones(
                                    (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                                ),
                                causal_mask,
                            ],
                            axis=-1,
                        )

                    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                else:
                    extended_attention_mask = attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                        input_shape, attention_mask.shape
                    )
                )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask):
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (:obj:`torch.Tensor`): An attention mask.

        Returns:
            :obj:`torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        if self.dtype == torch.float16:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        elif self.dtype == torch.float32:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            raise ValueError(
                "{} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`".format(
                    self.dtype
                )
            )

        return encoder_extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked = False):
        """
        Prepare the head mask if needed.

        Args:
            head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (:obj:`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (:obj:`bool`, `optional, defaults to :obj:`False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
            list with :obj:`[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    #Need raw input_ids
    def forward(self, input_ids=None, attention_mask = None, token_type_ids = None, position_ids = None, head_mask =None, inputs_embeds = None,
    encoder_hidden_states = None,encoder_attention_mask = None, output_attentions = None, output_hidden_states = None,joint=False):
        if joint:
            pair_ids = input_ids[1]
            input_ids = input_ids[0]
            pair_mask = attention_mask[1].to(input_ids.device)
            attention_mask = attention_mask[0].to(input_ids.device)
            token_type_ids = torch.zeros(input_ids.size(), device=input_ids.device if input_ids is not None else inputs_embeds.device).long()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()[:-1]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape,device = device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape,dtype=torch.float,device=device)
        #attention_mask = torch.tensor(attention_mask,dtype=float)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask,input_shape,device,joint)

        if joint:
            extended_attention_mask2: torch.Tensor = self.get_extended_attention_mask(pair_mask,input_shape,device,joint)
            extended_attention_mask = torch.cat((extended_attention_mask,extended_attention_mask2),dim=-1)


        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, dtype=torch.float, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask,self.config.num_hidden_layers)

        embedding_output = self.embeddings(
                    input_ids=input_ids.long(), position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
            
        if joint:
            embedding_output = self.jointEmbeddings(embedding_output, pair_ids)

        encoder_outputs =self.encoder(
            embedding_output,
            attention_mask = extended_attention_mask,
            head_mask = head_mask,
            encoder_hidden_states = encoder_hidden_states,
            encoder_attention_mask = encoder_extended_attention_mask,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output,pooled_output,) + encoder_outputs[
            1:
        ]
        
        return outputs

class MMBertPreTrainingHeads(BertPreTrainingHeads):
    def __init__(self,config):
        super().__init__(config)
        self.align = torch.nn.Linear(config.hidden_size,2)

    def forward(self,sequence_output,pooled_output,joint=False):
        predictionScores = self.predictions(sequence_output)

        if joint:
            #Is it True?
            firstTokenTensor = sequence_output[:,0]
            alignScore = self.align(firstTokenTensor)
            return predictionScores,alignScore
        else:
            seqRelationScore = self.seq_relationship(pooled_output)
            return predictionScores,seqRelationScore

class MMBertForPretraining(BertForPreTraining):
    def __init__(self,config):
        super().__init__(config)
        self.cls = MMBertPreTrainingHeads(config)
        self.bert = MMBertModel(config)
        self.num_labels = config._num_labels
        self.classifier1_1 = nn.Linear(config.hidden_size*3,config.hidden_size*1)
        if self.num_labels == 7:
            self.classifier1_2 = nn.Linear(config.hidden_size*1, 1)
        else:
            self.classifier1_2 = nn.Linear(config.hidden_size*1,self.num_labels)
        self.attn = nn.Linear(config.hidden_size*2,config.hidden_size*1)
        self.relu = nn.ReLU()
        self.v = nn.Linear(config.hidden_size, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        #?
        self.init_weights()
        
    def get_bert_output(self, input_ids, attention_mask, token_type_ids, joint = False):
        if joint:
            assert isinstance(input_ids,tuple)
        # 수정해야 함
        outputs = self.bert(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            joint = joint,
        )
        
        sequence_output, pooled_output = outputs[:2]
        return self.cls(sequence_output,pooled_output,joint), pooled_output

    def get_outputs(self, input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_label, joint=False):
        if input_ids is not None:
            (prediction_scores, relationship_score), pooled_output = self.get_bert_output(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                joint = joint
            )
            self.outputs += (prediction_scores, relationship_score)

            total_loss = 0.0

            loss_fct = torch.nn.CrossEntropyLoss()
            if masked_lm_labels is not None:
                masked_lm_loss = loss_fct(prediction_scores.view(-1,self.config.vocab_size), masked_lm_labels.view(-1))
                total_loss += masked_lm_loss

            if next_sentence_label is not None:
                next_sentence_loss = loss_fct(relationship_score.view(-1,2),next_sentence_label.view(-1))
                total_loss += next_sentence_loss

        return total_loss, pooled_output
    
    def forward(self, input_ids, token_type_ids, attention_mask, masked_labels, ap_label, sentiment):
        self.outputs = ()
        text_loss, visual_loss, speech_loss = None, None, None
        text_pooled_output, visual_pooled_output, speech_pooled_output = None, None, None
        text_input_ids, visual_input_ids, speech_input_ids, text_with_visual_ids, text_with_speech_ids = input_ids
        text_token_type_ids, visual_token_type_ids, speech_token_type_ids = token_type_ids
        text_attention_mask, visual_attention_mask, speech_attention_mask = attention_mask
        text_masked_lm_labels, visual_masked_lm_labels, speech_masked_lm_labels = masked_labels
        visual_next_sentence_label, speech_next_sentence_label = ap_label

        text_loss, text_pooled_output = self.get_outputs(text_input_ids, text_attention_mask, text_token_type_ids, text_masked_lm_labels, None, False)
        visual_loss, visual_pooled_output = self.get_outputs((text_with_visual_ids, visual_input_ids), visual_attention_mask, visual_token_type_ids, visual_masked_lm_labels, visual_next_sentence_label, True)
        speech_loss, speech_pooled_output = self.get_outputs((text_with_speech_ids, speech_input_ids), speech_attention_mask, speech_token_type_ids, speech_masked_lm_labels, speech_next_sentence_label, True)

        if text_pooled_output is not None and visual_pooled_output is not None and speech_pooled_output is not None:
            text_pooled_score = self.v(self.relu(self.attn(torch.cat((text_pooled_output, text_pooled_output),dim=1))))
            visual_pooled_score = self.v(self.relu(self.attn(torch.cat((visual_pooled_output, visual_pooled_output),dim=1))))
            speech_pooled_score = self.v(self.relu(self.attn(torch.cat((speech_pooled_output, speech_pooled_output),dim=1))))
            text_pooled_output = (text_pooled_output * text_pooled_score)
            visual_pooled_output = (visual_pooled_output * visual_pooled_score)
            speech_pooled_output = (speech_pooled_output * speech_pooled_score)
            pooled_output = torch.cat((text_pooled_output,visual_pooled_output,speech_pooled_output),dim=1)
            temp = self.classifier1_1(pooled_output)
            logits = self.classifier1_2(temp)
            mlm_loss = (text_loss + visual_loss + speech_loss)/3.0

        if sentiment is not None:
            if self.num_labels == 1 or self.num_labels == 7:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    logits = self.tanh(logits)
                label_loss = loss_fct(logits.view(-1), sentiment.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                label_loss = loss_fct(
                    logits, sentiment
                    )
                logits = torch.argmax(self.sigmoid(logits),dim=1)

        joint_loss = mlm_loss + label_loss
        self.outputs =  (joint_loss, text_loss, visual_loss, speech_loss, label_loss,) + self.outputs
        #outputs = None

        return self.outputs, logits