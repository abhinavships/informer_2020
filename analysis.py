from models.model import Informer
import torch
#for getting info of the mdoel)

model = Informer(enc_in=7, dec_in=7, c_out=7, seq_len=96, label_len=48, out_len=24, factor=5, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', output_attention=False, distil=True)
print(model)

# accessingt the weights of the query projection layer in the first attention layer of the encoder
W_query = model.encoder.attn_layers[0].attention.query_projection.weight
print(W_query.shape)


import numpy as np

W_numpy = W_query.detach().numpy()
U, S, Vh = np.linalg.svd(W_numpy)
print(S)



import matplotlib.pyplot as plt

plt.figure()
plt.plot(S)
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('Singular Value Spectrum - Query Projection - Encoder Layer 0')
plt.savefig('sv_spectrum_query.png')
plt.show()



