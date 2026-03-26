from models.model import Informer
import torch
#for getting info of the mdoel)

model = Informer(enc_in=7, dec_in=7, c_out=7, seq_len=96, label_len=48, out_len=24, factor=5, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', output_attention=False, distil=True)
print(model)


W_query = model.encoder.attn_layers[0].attention.query_projection.weight
print(W_query.shape)