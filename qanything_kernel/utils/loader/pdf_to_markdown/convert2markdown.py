import json
import re


def json2markdown(json_dir, markdown_dir):
    pdf = []
    para_before = ''
    before_patt = ['[a-z]', ',', '，', '-', '\(', '\)', '（', '）']
    after_patt = ['[a-z]', '·', '\(', '\)', '（', '）']
    js = json_dir 
    f = json.load(open(js))
    md = markdown_dir
    for i, section in f.items():
        text = section['text']
        type = section['type']
        if isinstance(text, dict):
            table = text['table_markdown']
            caption = text['table_caption']
            pdf.append(table)
            pdf.append(caption + '\n\n')
        elif type.startswith('title'):
            pdf.append('## '+text.split('@@')[0])
        elif 'figure' in type:
            pdf.append('![figure]'+'({}.jpg "{}")'.format(type, text) + '\n')
        elif 'equation' in type:
            pdf.append('![equation]'+'({}.jpg)'.format(type) + '\n')
        else:
            para = text.split('@@')[0] + '\n'
            if para=="\n":
                continue
            pdf.append(para)
            if para_before:
                if any([re.match(p, para_before[-2]) for p in before_patt]) and any([re.match(p, para[0]) for p in after_patt]):
                    pdf.pop(pdf.index(para))
                    pdf.pop(pdf.index(para_before))
                    pdf.append(para_before[:-1] + ' ' + para)
                    para_before = para_before[:-1] + ' ' + para
                else:
                    para_before = para
            else:
                para_before = para
    # print(pdf, flush=True)
    for p in pdf:
        print(p, file=open(md, 'a'))
    return '\n'.join(pdf)
        
        


if __name__ =='__main__':
    json_dir = 'QANY_DB/content/tsh/KB1234/86996323ae634c4eb1f598ad3337487e/成都市残疾儿童康复救助工作实施办法.json'
    json2markdown(json_dir, "./markdown_dir.md")