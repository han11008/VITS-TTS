import argparse, json, text, pydub, os
from utils import load_filepaths_and_text
from random import shuffle
from glob import glob
from collections import defaultdict

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--max_epochs', type=int, default=500,
                      help='Max epochs for training')

  args = parser.parse_args()

  with open('configs/ljs_base.json', 'r') as f:
    hps = json.load(f)

  hps['train']['log_interval'] = 10
  hps['train']['eval_interval'] = 100
  hps['train']['batch_size'] = 20 # 12G RAM
  hps['train']['epochs'] = args.max_epochs
  hps['data']['training_files'] = 'filelists/train.txt'
  hps['data']['validation_files'] = 'filelists/valid.txt'
  hps['data']['text_cleaners'] = ['zh_en_cleaners']

  with open('configs/modified.json', 'w', encoding='utf-8') as f:
    json.dump(hps, f, indent=2)

  audios = defaultdict(bool)
  for audio in glob('audios/*'):
    audios[audio] = True

  data_path = 'filelists/annotations.txt'
  if not os.path.exists(data_path):
    raise SystemExit(f'{data_path} does not exist!')
  train_data = load_filepaths_and_text(data_path)
  print(f'Number of annotated data: {len(train_data)}')
  total_audio_len = 0
  print('-----\nMissing or improper audio file(s):')
  for i in range(len(train_data)):
    audio_file, original_text = train_data[i]
    if audios[audio_file]:
      cleaned_text = text._clean_text(original_text, hps['data']['text_cleaners'])
      train_data[i][1] = cleaned_text

      audio_data = pydub.AudioSegment.from_wav(audio_file)
      misformat_log = ''
      if audio_data.channels != 1:
        misformat_log += f'channels={audio_data.channels}'
      if audio_data.sample_width != 2:
        sep = ', ' if misformat_log else ''
        misformat_log += f'{sep}data_width={audio_data.sample_width}'
      if audio_data.frame_rate != 22050:
        sep = ', ' if misformat_log else ''
        misformat_log += f'{sep}sample_rate={audio_data.frame_rate}'

      if misformat_log:
        print(f'\t{audio_file}:\t{misformat_log}')
      else:
        total_audio_len += audio_data.duration_seconds
    else:
      print(f'\t{audio_file}:\tnot exist')
  print('-----')

  shuffle(train_data)
  valid_num = round(len(train_data) * 0.15)
  valid_data = train_data[-valid_num:]
  train_data = train_data[:-valid_num]

  print(f'Number of training data: {len(train_data)}')
  print(f'Number of validation data: {len(valid_data)}')
  print(f'Total audio length in seconds: {total_audio_len}')

  with open('filelists/train.txt', 'w', encoding='utf-8') as f:
    f.writelines(["|".join(x) + "\n" for x in train_data])
  with open('filelists/valid.txt', 'w', encoding='utf-8') as f:
    f.writelines(["|".join(x) + "\n" for x in valid_data])

