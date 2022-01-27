
import numpy as np
from nltk.tokenize import RegexpTokenizer

word_tokenizer = RegexpTokenizer(
        r'''(?x)          # set flag to allow verbose regexps
            (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A. or U.S.A #
          | (?:\d+\.)           # numbers
          | \w+(?:[-.]\w+)*     # words with optional internal hyphens
          | \$?\d+(?:.\d+)?%?   # currency and percentages, e.g. $12.40, 82%
          | \.\.\.              # ellipsis, and special chars below, includes ], [
          | [-\]\[.,;"'?():_`“”/°º‘’″…#$%()*+<>=@\\^_{}|~❑&§]
        '''
        )

def convert_to_word(text, token_output, token_spans):
    word_spans = list(word_tokenizer.span_tokenize(text))
    word_start_locs, word_end_locs = zip(*word_spans)
    word_output = len(word_start_locs)*[0]
    for i, (word_start, word_end) in enumerate(word_spans):
        for j, (token_start, token_end) in enumerate(token_spans):
            if token_start == token_end: 
                continue
            elif word_start == token_start and word_end >= token_end:
                word_output[i] = token_output[j]
    return word_spans, word_output

def io_decoder(text: str, output: list, spans: list, tag_names: list):
    word_spans, word_output = convert_to_word(text, output, spans)
    entities = []
    tag_pre = 'O'
    for tag_idx, (s,e) in zip(word_output, word_spans):
        tag = tag_names[tag_idx]
        if tag != 'O' and tag_pre != tag: # start of new entity
            entities.append({'type': tag[2:], 'found': [s, e]})
        elif tag != 'O' and tag_pre == tag: # continuation of entity, extend the entity
                entities[-1]['found'][1] = e
        # else: tag is 'O', therefore no more conditions..
        #	continue
        tag_pre = tag
    # add text spans to dictionary
    for entity_ in entities:
        entity_['text'] = text[entity_['found'][0]:entity_['found'][1]]
    return entities

def bio_decoder(text: str, output: list, spans: list, tag_names: list):
    word_spans, word_output = convert_to_word(text, output, spans)
    entities = []
    tag_pre = 'O'
    for tag_idx, (s,e) in zip(word_output, word_spans):
        tag = tag_names[tag_idx]
        if tag.startswith('B-'):
            entities.append({'entity': tag[2:], 'start': s, 'end': e})
        elif tag.startswith('I-'):
            if tag_pre == 'O' or tag_pre[2:] != tag[2:]:
                tag = 'O' # not logical, so assign O to this token.
            else:
                entities[-1]['end'] = e
        # else: tag is 'O', therefore no more conditions..
        tag_pre = tag
    # add text spans to dictionary
    for entity_ in entities:
        entity_['text'] = text[entity_['start']:entity_['end']]
    return entities

decoder_mapping = {
    "io": io_decoder,
    "bio": bio_decoder
}
