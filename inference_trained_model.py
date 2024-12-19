import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os, re, sys, torch
from torch import nn, LongTensor

import commons, utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write as write_wav

def get_text(text, hps):
  text_norm = text_to_sequence(text, hps.data.text_cleaners)
  if hps.data.add_blank:
    text_norm = commons.intersperse(text_norm, 0)
  text_norm = LongTensor(text_norm)
  return text_norm

def inference(text):
  config_path = 'logs/config.json'
  try:
    model_path = utils.latest_checkpoint_path('logs', 'G_*.pth')
  except:
    model_path = ''
  if not (os.path.exists(config_path) and os.path.exists(model_path)):
    raise SystemExit('There are no trained models.')

  hps = utils.get_hparams_from_file(config_path)
  net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
  _ = net_g.eval()
  _ = utils.load_checkpoint(model_path, net_g, None)

  stn_tst = get_text(text, hps)
  with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

  audio_path = 'logs/test.wav'
  write_wav(audio_path, hps.data.sampling_rate, audio)
  print(f'The audio of text "{text}" is saved at {audio_path}(using model {model_path})')

if __name__ == '__main__':
  if len(sys.argv) > 1:
    content = ' '.join(sys.argv[1:])
    inference(content)
  else:
    print('Example usage: `python3 inference_trained_model.py "Hello, world!"`')
