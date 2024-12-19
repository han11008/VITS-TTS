""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re, itertools
from unidecode import unidecode
from phonemizer.backend import EspeakBackend
from logging import ERROR
en_phonemizer = EspeakBackend('en-us', preserve_punctuation=True)
en_phonemizer.logger.setLevel(ERROR)

import os
os.environ['HF_HOME'] = os.environ['HOME'] + '/hf_home'
from g2pw import G2PWConverter
zh_converter = G2PWConverter()

EN_MASK = 'X'
EN_PRESERVE = []

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)


def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def english_cleaners(text):
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  phonemes = en_phonemizer.phonemize([text], strip=True)[0]
  phonemes = collapse_whitespace(phonemes)
  return phonemes


def search_en(match):
  global EN_MASK, EN_PRESERVE
  EN_PRESERVE.append(match.group(0))
  return EN_MASK

def place_back_en(match):
  global EN_PRESERVE
  text = EN_PRESERVE.pop(0)
  return f' {english_cleaners(text)} '

def zh_en_cleaners(text):
  global EN_MASK, EN_PRESERVE
  EN_PRESERVE = []
  text = mandarin_punct_modify(text)
  text = re.sub(r'[a-zA-Z]+[^\u4e00-\u9fff]*', search_en, text)
  text = mandarin_cleaners(text)
  text = re.sub(EN_MASK, place_back_en, text)
  return text.strip()


_bopomofo_to_ipa = [(re.compile('%s' % x[0]), x[1]) for x in [
    ('ㄅㄛ', 'BUO'),
    ('ㄅ', 'ʙ'),
    ('ㄆㄛ', 'PUO'),
    ('ㄆ', 'P'),
    ('ㄇㄛ', 'MUO'),
    ('ㄇ', 'M'),
    ('ㄈㄛ', 'FUO'),
    ('ㄈ', 'F'),
    ('ㄉ', 'D'),
    ('ㄊ', 'T'),
    ('ㄋ', 'N'),
    ('ㄌ', 'L'),
    ('ㄍ', 'g'),
    ('ㄎ', 'k'),
    ('ㄏ', 'h'),
    ('ㄐ', 'ɮ'),
    ('ㄑ', 'tɕ'),
    ('ㄒ', 'ɕ'),
    ('ㄓ', 'ʤ'),
    ('ㄔ', 'tʰʂ'),
    ('ㄕ', 'ʂ'),
    ('ㄖ', 'ʐ'),
    ('ㄗ', 'Z'),
    ('ㄘ', 'ʧ'),
    ('ㄙ', 's'),
    ('ㄧㄚ', 'ja'),
    ('ㄧㄛ', 'jo'),
    ('ㄧㄞ', 'jaɪ'),
    ('ㄧㄝ', 'jæ'),
    ('ㄧㄠ', 'jaʊ'),
    ('ㄧㄡ', 'joʊ'),
    ('ㄧㄢ', 'jen'),
    ('ㄧㄣ', 'jin'),
    ('ㄧㄤ', 'jaŋ'),
    ('ㄧㄥ', 'jəŋ'),
    ('ㄧ', 'i'),
    ('ㄨㄚ', 'wa'),
    ('ㄨㄛ', 'wo'),
    ('ㄨㄞ', 'waɪ'),
    ('ㄨㄟ', 'we'),
    ('ㄨㄢ', 'wen'),
    ('ㄨㄣ', 'wun'),
    ('ㄨㄤ', 'waŋ'),
    ('ㄨㄥ', 'woŋ'),
    ('ㄨ', 'u'),
    ('ㄩㄝ', 'ɥæ'),
    ('ㄩㄢ', 'ɥaŋ'),
    ('ㄩㄣ', 'ɥn'),
    ('ㄩㄥ', 'ɥŋ'),
    ('ㄩ', 'ʉ'),
    ('ㄚ', 'a'),
    ('ㄛ', 'o'),
    ('ㄜ', 'œ'),
    ('ㄝ', 'æ'),
    ('ㄞ', 'aɪ'),
    ('ㄟ', 'e'),
    ('ㄠ', 'aʊ'),
    ('ㄡ', 'oʊ'),
    ('ㄢ', 'an'),
    ('ㄣ', 'ʌn'),
    ('ㄤ', 'aŋ'),
    ('ㄥ', 'oŋ'),
    ('ㄦ', 'ɝ'),
    ('1', '→'),
    ('2', '↑'),
    ('3', '↓↑'),
    ('4', '↓'),
    ('5', ' '),
    ('，', ','),
    ('。', '.'),
    ('！', '!'),
    ('？', '?'),
    ('—', '-')
]]

_measure_words = list('個把對張本冊筆齣部堆場段副道床串層頂次棟幢簇堵滴頓點朵疊發番方份杆峰掛幅行服戶桌夥伙尊級宗劑台臺家頭灘團堂尾輛席套股列記具枚棵門架節面孔首束排撇盞樣泓彎趟句縷根顆間名口粒件片塊株泡闋匹扇批聲篇乘絲幀艘陣所則爿綹雙枝條隻支項座')

# https://github.com/coqui-ai/TTS/blob/main/TTS/tts/utils/text/chinese_mandarin/numbers.py
def _num2chinese(num: str, big=False, simp=False, o=False, twoalt=True) -> str:
    """Convert numerical arabic numbers (0->9) to chinese hanzi numbers (〇 -> 九)

    Args:
        num (str): arabic number to convert
        big (bool, optional): use financial characters. Defaults to False.
        simp (bool, optional): use simplified characters instead of tradictional characters. Defaults to True.
        o (bool, optional): use 〇 for 'zero'. Defaults to False.
        twoalt (bool, optional): use 两/兩 for 'two' when appropriate. Defaults to False.

    Raises:
        ValueError: if number is more than 1e48
        ValueError: if 'e' exposent in number

    Returns:
        str: converted number as hanzi characters
    """

    # check num first
    nd = str(num)
    if abs(float(nd)) >= 1e48:
        raise ValueError("number out of range")
    if "e" in nd:
        raise ValueError("scientific notation is not supported")
    c_symbol = "正负点" if simp else "正負點"
    if o:  # formal
        twoalt = False
    if big:
        c_basic = "零壹贰叁肆伍陆柒捌玖" if simp else "零壹貳參肆伍陸柒捌玖"
        c_unit1 = "拾佰仟"
        c_twoalt = "贰" if simp else "貳"
    else:
        c_basic = "〇一二三四五六七八九" if o else "零一二三四五六七八九"
        c_unit1 = "十百千"
        if twoalt:
            c_twoalt = "两" if simp else "兩"
        else:
            c_twoalt = "二"
    c_unit2 = "万亿兆京垓秭穰沟涧正载" if simp else "萬億兆京垓秭穰溝澗正載"
    revuniq = lambda l: "".join(k for k, g in itertools.groupby(reversed(l)))
    nd = str(num)
    result = []
    if nd[0] == "+":
        result.append(c_symbol[0])
    elif nd[0] == "-":
        result.append(c_symbol[1])
    if "." in nd:
        integer, remainder = nd.lstrip("+-").split(".")
    else:
        integer, remainder = nd.lstrip("+-"), None
    if int(integer):
        splitted = [integer[max(i - 4, 0) : i] for i in range(len(integer), 0, -4)]
        intresult = []
        for nu, unit in enumerate(splitted):
            # special cases
            if int(unit) == 0:  # 0000
                intresult.append(c_basic[0])
                continue
            if nu > 0 and int(unit) == 2:  # 0002
                intresult.append(c_twoalt + c_unit2[nu - 1])
                continue
            ulist = []
            unit = unit.zfill(4)
            for nc, ch in enumerate(reversed(unit)):
                if ch == "0":
                    if ulist:  # ???0
                        ulist.append(c_basic[0])
                elif nc == 0:
                    ulist.append(c_basic[int(ch)])
                elif nc == 1 and ch == "1" and unit[1] == "0":
                    # special case for tens
                    # edit the 'elif' if you don't like
                    # 十四, 三千零十四, 三千三百一十四
                    ulist.append(c_unit1[0])
                elif nc > 1 and ch == "2":
                    ulist.append(c_twoalt + c_unit1[nc - 1])
                else:
                    ulist.append(c_basic[int(ch)] + c_unit1[nc - 1])
            ustr = revuniq(ulist)
            if nu == 0:
                intresult.append(ustr)
            else:
                intresult.append(ustr + c_unit2[nu - 1])
        result.append(revuniq(intresult).strip(c_basic[0]))
    else:
        result.append(c_basic[0])
    if remainder:
        result.append(c_symbol[2])
        result.append("".join(c_basic[int(ch)] for ch in remainder))
    return "".join(result)

def number_to_chinese(text):
    numbers = re.findall(r'\d+(?:\.?\d+)?', text)
    for number in numbers:
        n2c = _num2chinese(number)
        if n2c == '二':
            idx = text.find(number)
            if idx + 1 < len(text) and text[idx + 1] in _measure_words:
            	n2c = '兩'
        text = text.replace(number, n2c, 1)
    return text

def mandarin_to_ipa(text):
    pronunce = zh_converter(text)[0]
    for idx in range(len(pronunce)):
        if pronunce[idx] is None:
            pronunce[idx] = text[idx]
    text = ''.join(pronunce)
    for regex, replacement in _bopomofo_to_ipa:
        text = re.sub(regex, replacement, text)
    text = collapse_whitespace(text)
    return text

def mandarin_punct_modify(text):
    text = text.replace('，', ',')
    text = text.replace('、', ',')
    text = text.replace('。', '.')
    text = text.replace('：', ':')
    text = text.replace('；', ';')
    text = text.replace('·', '')
    text = text.replace('（', '(')
    text = text.replace('）', ')')
    text = text.replace('「', '"')
    text = text.replace('」', '"')
    text = text.replace('『', '"')
    text = text.replace('』', '"')
    text = text.replace('《', '"')
    text = text.replace('》', '"')
    return text

def mandarin_cleaners(text):
    text = number_to_chinese(text)
    text = mandarin_to_ipa(text)
    text = re.sub(r'(ʤ|tʰʂ|ʂ|ʐ|Z|ʧ|s)([→↑↓ ])', r'\1ɭ\2', text)
    return text
