from transformers import BertForPreTraining, BertModel
from transformers.modeling_bert import BertPreTrainingHeads
import torch

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

class MMBertForPreTraining(BertForPreTraining):
    def __init__(self,config):
        super().__init__(config)
        self.cls = MMBertPreTrainingHeads(config)
        self.bert = MMBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,config.num_labels)
        
        #?
        self.init_weights()
        
    def get_bert_outputs(self,input_ids,attention_mask,token_type_ids,joint = False):
        # 수정해야 함
        outputs = self.bert(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        
        sequence_output, pooled_output = outputs[:2]
        return self.cls(sequence_output,pooled_output,joint), pooled_output
    
    def forward(self,
                text_input_ids = None, visual_input_ids = None, speech_input_ids=None,
                text_token_type_ids = None, visual_token_type_ids = None, speech_token_type_ids = None,
                text_attention_maks = None, visual_attention_mask = None, speech_attention_mask = None,
                text_masked_lm_labels = None, visual_masked_lm_labels = None, speech_masked_lm_labels = None,
                text_next_sentence_label = None, visual_next_sentence_label = None, speech_masked_lm_labels = None
               ):
        outputs = ()
        text_loss = None
        visual_loss = None
        speech_loss = None
        
        if text_input_ids is not None:
            (text_prediction_scores, text_seq_relationship_score), text_pooled_output = self.get_bert_output(
                input_ids = text_input_ids,
                attention_mask = text_attention_mask,
                token_type_ids = text_token_type_ids,
            )
            
            outputs += (text_prediction_scores, text_seq_relationship_score)
            
            if text_masked_lm_labels is not None and text_next_sentence_label is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                masked_lm_loss = loss_fct(text_prediction_scores.view(-1,self.config.vocab_size),text_masked_lm_labels.view(-1))
                next_sentence_loss = loss_fct(text_seq_relationship_score.view(-1,2), text_next_sentence_label.view(-1))
                total_loss = masked_lm_loss + next_sentence_loss
                text_loss = total_loss
                
        if visual_input_ids is not None:
            (visual_prediction_scores, visual_seq_relationship_score), visual_pooled_output = self.get_bert_output(
                input_ids = visual_input_ids,
                attention_mask = visual_attention_mask,
                token_type_ids = visual_token_type_ids,
            )
            
            outputs += (visual_prediction_scores, visual_seq_relationship_score)
            
            if visual_masked_lm_labels is not None and visual_next_sentence_label is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                masked_lm_loss = loss_fct(visual_prediction_scores.view(-1,self.config.vocab_size),visual_masked_lm_labels.view(-1))
                next_sentence_loss = loss_fct(visual_seq_relationship_score.view(-1,2), visual_next_sentence_label.view(-1))
                total_loss = masked_lm_loss + next_sentence_loss
                visual_loss = total_loss
        
        if speech_input_ids is not None:
            (speech_prediction_scores, speech_seq_relationship_score), speech_pooled_output = self.get_bert_output(
                input_ids = speech_input_ids,
                attention_maks = speech_attention_mask,
                token_type_ids = speech_token_type_ids,
            )
            
            outputs += (speech_prediction_scores, speech_seq_relationship_score)
            
            if speech_masked_lm_labels is not None and speech_next_sentence_label is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                masked_lm_loss = loss_fct(speech_prediction_scores.view(-1,self.config.vocab_size), speech_masked_lm_labels.view(-1))
                next_sentence_loss = loss_fct(speech_seq_relationship_score.view(-1,2),speech_next_sentence_label.view(-1))
                total_loss = masked_lm_loss+next_sentence_loss
                speech_loss +=total_loss
        
        #Need Check
        pooled_output = torch.cat(text_pooled_output,visual_pooled_output,speech_pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = (logits,) + outputs
        
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
            
        return outputs