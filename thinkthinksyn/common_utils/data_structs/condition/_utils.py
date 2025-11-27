# Some codes are copy & modify from `https://github.com/nrchan/chinese-number-converter

import re
import hashlib

ChineseNumber2Number = {
    "1": 1,
    "一": 1,
    "壹": 1,
    "2": 2,
    "二": 2,
    "貳": 2,
    "貮": 2,
    "贰": 1,
    "兩": 2,
    "两": 2,
    "3": 3,
    "三": 3,
    "叁": 3,
    "參": 3,
    "叄": 3,
    "参": 3,
    "4": 4,
    "四": 4,
    "肆": 4,
    "5": 5,
    "五": 5,
    "伍": 5,
    "6": 6,
    "六": 6,
    "陸": 6,
    "陆": 6,
    "7": 7,
    "七": 7,
    "柒": 7,
    "8": 8,
    "八": 8,
    "捌": 8,
    "9": 9,
    "九": 9,
    "玖": 9,
    "0": 0,
    "零": 0,
    "〇": 0
}   
ChineseCombinedNumber2ChineseNumber = {
    "廿": "二十",
    "念": "二十",
    "卅": "三十",
    "卌": "四十",
    "圩": "五十",
    "圓": "六十",
    "圆": "六十",
    "進": "七十",
    "进": "七十",
    "枯": "八十",
    "樺": "九十",
    "桦": "九十",
    "皕": "兩百",
}
ChineseUnit2Number = {
    "十": 10,
    "拾": 10,
    "百": 100,
    "佰": 100,
    "千": 1000,
    "仟": 1000
}
ChineseBigUnit2Number = {
    "萬": 1e+4,
    "万": 1e+4,
    "億": 1e+8,
    "亿": 1e+8,
    "兆": 1e+12,
    "京": 1e+16,
    "垓": 1e+20,
    "秭": 1e+24,
    "穰": 1e+28,
    "溝": 1e+32,
    "沟": 1e+32,
    "澗": 1e+36,
    "涧": 1e+36,
    "正": 1e+40,
    "載": 1e+44,
    "载": 1e+44,
    "極": 1e+48,
    "极": 1e+48
}
ChineseUnit = [
    "",
    "十",
    "百",
    "千"
]
ChineseUnit_B = [
    "",
    "拾",
    "佰",
    "仟"
]
ChineseBigUnit = [
    "",
    "萬",
    "億",
    "兆",
    "京",
    "垓",
    "秭",
    "穰",
    "溝",
    "澗",
    "正",
    "載",
    "極"
]
ChineseBigUnit_S = [
    "",
    "万",
    "亿",
    "兆",
    "京",
    "垓",
    "秭",
    "穰",
    "沟",
    "涧",
    "正",
    "载",
    "极"
]
Number2Chinese = {
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
    0: "零"
}
Number2Chinese_B = {
    1: "壹",
    2: "貳",
    3: "參",
    4: "肆",
    5: "伍",
    6: "陸",
    7: "柒",
    8: "捌",
    9: "玖",
    0: "零"
}
Number2Chinese_S_B = {
    1: "壹",
    2: "贰",
    3: "叁",
    4: "肆",
    5: "伍",
    6: "陆",
    7: "柒",
    8: "捌",
    9: "玖",
    0: "零"
}

def chinese2number(string: str):
    #for 廿 and other combined number
    for combined, normal in ChineseCombinedNumber2ChineseNumber.items():
        string = string.replace(combined, normal)
    
    curDigit = 0
    curNum = 0
    num = 0
    for i in range(len(string)):
        c = string[i]
        if c in ChineseNumber2Number:
            curDigit *= 10
            curDigit += ChineseNumber2Number[c]

            #for "一百一" is 110 (not 101) issue
            if i == len(string)-1 and len(string) >= 2 and (string[i-1] == "百" or string[i-1] == "佰"):
                curDigit *= 10
            #for "一千一" is 1100 (not 1001) issue
            if i == len(string)-1 and len(string) >= 2 and (string[i-1] == "千" or string[i-1] == "仟"):
                curDigit *= 100
            #for "一萬一" is 11000 (not 10001) issue
            if i == len(string)-1 and len(string) >= 2 and (string[i-1] == "萬" or string[i-1] == "万"):
                curDigit *= 1000
        if c in ChineseUnit2Number:
            if curDigit == 0:
                curNum += ChineseUnit2Number[c]
            else:
                curNum += curDigit*ChineseUnit2Number[c]
            curDigit = 0
        if c in ChineseBigUnit2Number:
            curNum += curDigit
            curDigit = 0
            curNum *= ChineseBigUnit2Number[c]
            num += curNum
            curNum = 0
    curNum += curDigit
    num += curNum
    return num

_pattern = []
_pattern.extend(list(ChineseNumber2Number.keys()))
_pattern.extend(list(ChineseCombinedNumber2ChineseNumber.keys()))
_pattern.extend(list(ChineseUnit2Number.keys()))
_pattern.extend(list(ChineseBigUnit2Number.keys()))
_pattern.extend(ChineseUnit_B)
_pattern.extend(ChineseBigUnit)
_pattern.extend(ChineseBigUnit_S)
for p in tuple(_pattern):
    if isinstance(p, int) or \
        isinstance(p, str) and p.isdigit():  
        _pattern.remove(p)
_zh2num_within_text_pattern = "".join(_pattern)
_zh2num_within_text_pattern = re.compile(f"[{_zh2num_within_text_pattern}]+")

def chinese2number_within_text(text: str)->str:
    '''check any chinese number in the text and convert them to number,
    e.g. "我有一百塊" -> "我有100塊"'''
    return _zh2num_within_text_pattern.sub(lambda m: str(chinese2number(m.group(0))), text) # type: ignore

def get_num_from_text(s: str, raise_error: bool = False):
    '''
    Convert a string to a number. Supported formats:
    - Hexadecimal: '0x10' -> 16
    - Binary: '0b1010' -> 10
    - Octal: '0o12' -> 10
    - Decimal: '123.456' -> 123.456
    - Chinese numbers: '四百二十' -> 420
    '''
    if s.startswith('0x'):
        try:
            return int(s, 16)
        except:
            ...
    elif s.startswith('0b'):
        try:
            return int(s, 2)
        except:
            ...
    elif s.startswith('0o'):
        try:
            return int(s, 8)
        except:
            ...
    if s.isnumeric():
        try:
            return int(chinese2number_within_text(s))
        except:
            ...
    try:
        return float(s)
    except:
        ...
    if raise_error:
        raise ValueError(f'Cannot convert `{s}` to number')
    return None

def hash_md5(data: str) -> str:
    return hashlib.md5(data.encode()).hexdigest()


__all__ = [
    'chinese2number', 
    'chinese2number_within_text', 
    'get_num_from_text', 
    'hash_md5'
]


if __name__ == '__main__':
    print(chinese2number_within_text("一零"))
    print(chinese2number_within_text("0二"))
    print(chinese2number_within_text("二零二四十二零二"))
    print(chinese2number_within_text("二零二四一二零四"))