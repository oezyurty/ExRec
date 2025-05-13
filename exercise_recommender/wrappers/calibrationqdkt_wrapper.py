import torch
from pykt.models.qdkt import CalibrationQDKT
from exercise_recommender.utils.history_generator import HistoryGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

class CalibrationQDKTWrapper():
    def __init__(self,
                history_generator: HistoryGenerator,
                pretrained_model_path = "",
                total_batch_size: int=1024):
        
        if pretrained_model_path == "":
            raise("pretrained_model cannot be empty")
        
        if not history_generator:
            raise("HistoryGenerator object should be given to CalibrationQDKTWrapper.")

        self.pretrained_model = CalibrationQDKT()
        self.pretrained_model = self.pretrained_model.to(device)
        net = torch.load(pretrained_model_path, map_location=device)
        print(f"Pretrained model mapped device: {device}")
        self.pretrained_model.load_state_dict(net)
        self.pretrained_model.model.model.eval()
        
        self.last_h = torch.rand(self._hidden_size()).to(device)
        self.last_c = torch.rand(self._hidden_size()).to(device)
        self.last_lstm_out = torch.rand(self._hidden_size()).to(device)

        self.history = history_generator
        self.total_batch_size = total_batch_size
    
    def init_states(self, seq_size):
        """
        Creates last_h, last_c and last_lstm_out from the history generator.
        last_h has dimensions [1, total_batch_size, hidden_dim]
        last_c has dimensions [1, total_batch_size, hidden_dim]
        last_lstm_out has dimensions [total_batch_size, hidden_dim]
        """
        total_batch = 0
        all_last_lstm_out = []
        all_last_h = []
        all_last_c = []
        all_qid_seqs = []

        while total_batch < self.total_batch_size:
            qseqs, rseqs, qid_seqs = self.history.get_question_response_pairs()
            qseqs = qseqs[:, :seq_size]
            rseqs = rseqs[:, :seq_size]
            qid_seqs = qid_seqs[:, :seq_size]
            batch = qseqs.shape[0]
            total_batch += batch
            correctness_encoding = self.pretrained_model.model.model.correctness_encoding(rseqs.long())
            x = qseqs + correctness_encoding
            lstm_out, (h_last, c_last) = self.pretrained_model.model.model.lstm_layer(x)
            last_lstm_out = lstm_out[:, -1, :]
            all_last_lstm_out.append(last_lstm_out)
            all_last_h.append(h_last)
            all_last_c.append(c_last)
            all_qid_seqs.append(qid_seqs)
        all_qid_seqs = torch.cat(all_qid_seqs, dim=0).to(device)
        all_last_lstm_out = torch.cat(all_last_lstm_out, dim=0).to(device)
        all_last_h = torch.cat(all_last_h, dim=1).to(device) # dim=1 because each batch has dimension 1,64,300
        all_last_c = torch.cat(all_last_c, dim=1).to(device) # dim=1 because each batch has dimension 1,64,300
        if total_batch > self.total_batch_size:
            # Need to cut if the cut off data is added on top of total_batch_size: happens at the end of the circular dataloader
            all_qid_seqs = all_qid_seqs[:self.total_batch_size]
            all_last_lstm_out = all_last_lstm_out[:self.total_batch_size]
            all_last_h = all_last_h[:, :self.total_batch_size]
            all_last_c = all_last_c[:, :self.total_batch_size]
        self.last_lstm_out = all_last_lstm_out
        self.last_h = all_last_h
        self.last_c = all_last_c
        self._initial_h = all_last_h
        self._initial_c = all_last_c
        self._initial_lstm = all_last_lstm_out
        return self.last_lstm_out, all_qid_seqs
    
    def init_states_with_data(self, mini_batch_size, seq_size, qseqs, rseqs, qid_seqs):
        
        total_processed = 0
        all_last_lstm_out = []
        all_last_h = []
        all_last_c = []
        all_qid_seqs = []

        while total_processed < self.total_batch_size:
            _qseqs = qseqs[total_processed:total_processed+mini_batch_size, :seq_size]
            _rseqs = rseqs[total_processed:total_processed+mini_batch_size, :seq_size]
            _qid_seqs = qid_seqs[total_processed:total_processed+mini_batch_size, :seq_size]
            total_processed += mini_batch_size
            correctness_encoding = self.pretrained_model.model.model.correctness_encoding(_rseqs.long())
            x = _qseqs + correctness_encoding
            lstm_out, (h_last, c_last) = self.pretrained_model.model.model.lstm_layer(x)
            last_lstm_out = lstm_out[:, -1, :]
            all_qid_seqs.append(_qid_seqs)
            all_last_lstm_out.append(last_lstm_out)
            all_last_h.append(h_last)
            all_last_c.append(c_last)
        
        all_qid_seqs = torch.cat(all_qid_seqs, dim=0).to(device)
        all_last_lstm_out = torch.cat(all_last_lstm_out, dim=0).to(device)
        all_last_h = torch.cat(all_last_h, dim=1).to(device) # dim=1 because each batch has dimension 1,64,300
        all_last_c = torch.cat(all_last_c, dim=1).to(device) # dim=1 because each batch has dimension 1,64,300
        self.last_lstm_out = all_last_lstm_out
        self.last_h = all_last_h
        self.last_c = all_last_c
        self._initial_h = all_last_h
        self._initial_c = all_last_c
        self._initial_lstm = all_last_lstm_out
        return self.last_lstm_out, all_qid_seqs

    def update_hidden_state(self, questions, responses):
        # Shape of input questions: [batch_size, seq_len, emb_size]
        # Shape of input responses: [batch_size, seq_len]
        questions = questions.to(device)
        responses = responses.to(device)
        correctness_encoding = self.pretrained_model.model.model.correctness_encoding(responses.long().to(device))
        x = questions + correctness_encoding
        lstm_out, (h_last, c_last) = self.pretrained_model.model.model.lstm_layer(x, (self.last_h, self.last_c))
        self.last_h = h_last
        self.last_c = c_last
        self.last_lstm_out = lstm_out[:, -1, :]
        return self.last_lstm_out
    
    def predict_in_rl(self, new_q):
        # new_q should have shape [batch_size, emb_size]
        new_q = new_q.to(device)
        probs = self.pretrained_model.model.model.prediction_layer(self.last_lstm_out, new_q)
        return probs
    
    def reset_in_rl(self):
        self.last_h = self._initial_h
        self.last_c = self._initial_c
        self.last_lstm_out = self._initial_lstm
    
    def _hidden_size(self):
        return self.pretrained_model.model.model.hidden_size