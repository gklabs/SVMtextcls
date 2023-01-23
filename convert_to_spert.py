import os
import glob
from pathlib import Path
import logging
import tqdm
import re
from collections import OrderedDict, Counter
import spacy
import json
import torch
import numpy as np
import pickle
import pandas as pd
import constants as C
import ann as A
import string


TEXT_FILE_EXT = 'txt'
ANN_FILE_EXT = 'ann'
EVENT = "event"
RELATION = "relation"
TEXTBOUND = "textbound"
ATTRIBUTE = "attribute"
ARGUMENT = 'argument'
ENCODING = 'utf-8'

COMMENT_RE = re.compile(r'^#')
TEXTBOUND_RE = re.compile(r'^T\d+')
EVENT_RE = re.compile(r'^E\d+\t')
ATTRIBUTE_RE = re.compile(r'^A\d+\t')
RELATION_RE = re.compile(r'^R\d+\t')
RELATION_DEFAULT = 'relation'


SPERT_ID = "id"
SPERT_TOKENS = "tokens"
SPERT_ENTITIES = "entities"
SPERT_RELATIONS = "relations"
SPERT_SUBTYPES = "subtypes"
SPERT_TYPE = "type"
SPERT_START = "start"
SPERT_END = "end"
SPERT_HEAD = "head"
SPERT_TAIL = "tail"
SPERT_OFFSETS = "offsets"




def get_filename(path):
    root, ext = os.path.splitext(path)
    return root

def filename_check(fn1, fn2):
    '''
    Confirm filenames, regardless of directory or extension, match
    '''
    fn1 = get_filename(fn1)
    fn2 = get_filename(fn2)

    return fn1==fn2

def get_files(path, ext='.', relative=False):
    files = list(Path(path).glob('**/*.{}'.format(ext)))

    if relative:
        files = [os.path.relpath(f, path) for f in files]

    return files

def get_brat_files(path):
    '''
    Find text and annotation files
    '''
    # Text and annotation files
    text_files = get_files(path, TEXT_FILE_EXT, relative=False)
    ann_files = get_files(path, ANN_FILE_EXT, relative=False)

    # Check number of text and annotation files
    n_txt = len(text_files)
    n_ann = len(ann_files)
    msg = f'Number of text and annotation files do not match: {n_txt} txt VS {n_ann} ann'
    assert n_txt == n_ann, msg

    # Sort files
    text_files.sort()
    ann_files.sort()

    # Check the text and annotation filenames
    mismatches = [str((t, a)) for t, a in zip(text_files, ann_files) \
                                           if not filename_check(t, a)]
    fn_check = len(mismatches) == 0
    assert fn_check, '''txt and ann filenames do not match:\n{}'''. \
                        format("\n".join(mismatches))

    return (text_files, ann_files)

def import_brat_dir(path, sample_count=None):

    text_files, ann_files = get_brat_files(path)
    file_list = list(zip(text_files, ann_files))
    file_list.sort(key=lambda x: x[1])

    if (sample_count is not None) and (sample_count != -1):
        file_list = file_list[0:sample_count]
        logging.warn(f"-"*80)
        logging.warn(f"Truncating loaded files to first {sample_count} files")
        logging.warn(f"-"*80)

    logging.info(f"")
    logging.info(f"Importing brat directory: {path}")
    pbar = tqdm.tqdm(total=len(file_list))

    # Loop on annotated files
    out = []
    for fn_txt, fn_ann in file_list:

        # Use filename as ID
        id = os.path.splitext(os.path.relpath(fn_txt, path))[0]

        # Read text file
        with open(fn_txt, 'r', encoding= ENCODING) as f:
            text = f.read()

        # Read annotation file
        with open(fn_ann, 'r', encoding= ENCODING) as f:
            ann = f.read()

        out.append((id, text, ann))

        pbar.update(1)
    pbar.close()

    return out


def get_unique_arg(argument, arguments):

    if argument in arguments:
        argument_strip = argument.rstrip(string.digits)
        for i in range(1, 500):
            argument_new = f'{argument_strip}{i}'
            if argument_new not in arguments:
                break
    else:
        argument_new = argument

    assert argument_new not in arguments, "Could not modify argument for uniqueness"

    if argument_new != argument:
        #logging.warn(f"Event decoding: '{argument}' --> '{argument_new}'")
        pass

    return argument_new


def parse_textbounds(lines):
    """
    Parse textbound annotations in input, returning a list of
    Textbound.

    ex.
        T1	Status 21 29	does not
        T1	Status 27 30	non
        T8	Drug 91 99	drug use

    """

    textbounds = {}
    for l in lines:
        if TEXTBOUND_RE.search(l):

            # Split line
            id, type_start_end, text = l.split('\t', maxsplit=2)

            # Check to see if text bound spans multiple sentences
            mult_sent = len(type_start_end.split(';')) > 1

            # Multiple sentence span, only use portion from first sentence
            if mult_sent:

                # type_start_end = 'Drug 99 111;112 123'

                # type_start_end = ['Drug', '99', '111;112', '123']
                type_start_end = type_start_end.split()

                # type = 'Drug'
                # start_end = ['99', '111;112', '123']
                type_ = type_start_end[0]
                start_end = type_start_end[1:]

                # start_end = '99 111;112 123'
                start_end = ' '.join(start_end)

                # start_ends = ['99 111', '112 123']
                start_ends = start_end.split(';')

                # start_ends = [('99', '111'), ('112', '123')]
                start_ends = [tuple(start_end.split()) for start_end in start_ends]

                # start_ends = [(99, 111), (112, 123)]
                start_ends = [(int(start), int(end)) for (start, end) in start_ends]

                start = start_ends[0][0]

                # ends = [111, 123]
                ends = [end for (start, end) in start_ends]

                text = list(text)
                for end in ends[:-1]:
                    n = end - start
                    assert text[n].isspace()
                    text[n] = '\n'
                text = ''.join(text)

                start = start_ends[0][0]
                end = start_ends[-1][-1]

            else:
                # Split type and offsets
                type_, start, end = type_start_end.split()

            # Convert start and stop indices to integer
            start, end = int(start), int(end)

            # Build text bound object
            assert id not in textbounds
            textbounds[id] = A.Textbound(
                          id = id,
                          type_= type_,
                          start = start,
                          end = end,
                          text = text,
                          )

    return textbounds


def parse_attributes(lines):
    """
    Parse attributes, returning a list of Textbound.
        Assume all attributes are 'Value'

        ex.

        A2      Value T4 current
        A3      Value T11 none

    """

    attributes = {}
    for l in lines:

        if ATTRIBUTE_RE.search(l):

            # Split on tabs
            attr_id, attr_textbound_value = l.split('\t')

            if len(attr_textbound_value.split()) == 3:

                type, tb_id, value = attr_textbound_value.split()

            elif len(attr_textbound_value.split()) == 2:
                type, tb_id = attr_textbound_value.split()
                value = C.UNDETERMINED
            else:
                raise ValueError(f"Invalid textbound attribute: repr({l})")



            # Add attribute to dictionary
            if tb_id in attributes:
                logging.warn(f'{tb_id} in {attributes.keys()}')
                raise ValueError("tb id already exists in dictionary")

            attributes[tb_id] = A.Attribute( \
                    id = attr_id,
                    type_ = type,
                    textbound = tb_id,
                    value = value)
    return attributes

def parse_events(lines):
    """
    Parse events, returning a list of Textbound.

    ex.
        E2      Tobacco:T7 State:T6 Amount:T8 Type:T9 ExposureHistory:T18 QuitHistory:T10
        E4      Occupation:T9 State:T12 Location:T10 Type:T11

        id     event:tb_id ROLE:TYPE ROLE:TYPE ROLE:TYPE ROLE:TYPE
    """

    events = {}
    for l in lines:
        if EVENT_RE.search(l):

            # Split based on white space
            entries = [tuple(x.split(':')) for x in l.split()]

            # Get ID
            id = entries.pop(0)[0]

            # Entity type
            event_type, _ = tuple(entries[0])

            # Role-type
            arguments = OrderedDict()
            for i, (argument, tb) in enumerate(entries):

                argument = get_unique_arg(argument, arguments)
                assert argument not in arguments
                arguments[argument] = tb

            # Only include desired arguments
            events[id] = A.Event( \
                      id = id,
                      type_ = event_type,
                      arguments = arguments)

    return events


def parse_relations(lines):
    """
    Parse events, returning a list of Textbound.

    ex.
    R1  attr Arg1:T2 Arg2:T1
    R2  attr Arg1:T5 Arg2:T6
    R3  attr Arg1:T7 Arg2:T1

    """

    relations = {}
    for line in lines:
        if RELATION_RE.search(line):

            # road move trailing white space
            line = line.rstrip()

            x = line.split()
            id = x.pop(0)
            role = x.pop(0)
            arg1 = x.pop(0).split(':')[1]
            arg2 = x.pop(0).split(':')[1]

            # Only include desired arguments
            assert id not in relations
            relations[id] = A.Relation( \
                      id = id,
                      role = role,
                      arg1 = arg1,
                      arg2 = arg2)

    return relations


def get_annotations(ann):
    '''
    Load annotations, including taxbounds, attributes, and events

    ann is a string
    '''

    # Parse string into nonblank lines
    lines = [l for l in ann.split('\n') if len(l) > 0]


    # Confirm all lines consumed
    remaining = [l for l in lines if not \
            ( \
                COMMENT_RE.search(l) or \
                TEXTBOUND_RE.search(l) or \
                EVENT_RE.search(l) or \
                RELATION_RE.search(l) or \
                ATTRIBUTE_RE.search(l)
            )
        ]
    msg = 'Could not match all annotation lines: {}'.format(remaining)
    assert len(remaining)==0, msg

    # Get events
    events = parse_events(lines)

    # Get relations
    relations = parse_relations(lines)

    # Get text bounds
    textbounds = parse_textbounds(lines)

    # Get attributes
    attributes = parse_attributes(lines)

    return (events, relations, textbounds, attributes)


def start_match(x_start, y_start, y_end):
    '''
    Determine if x is in range of y
    x_start:

    '''
    return (x_start >= y_start) and (x_start <  y_end)

def end_match(x_end, y_start, y_end):
    '''
    Determine if x is in range of y
    x_start:

    '''

    return (x_end   >  y_start) and (x_end   <= y_end)

def get_tb_indices(tb_dict, offsets):
    """
    Get sentence index for textbounds
    returns map in dictionary: [Textbound: (sentenceID, Token_start_index, Token_end_index)]
    e.g.
    'T1: (3,2,3)'
    """

    map = {}

    # iterate over text bounds
    for tb_id, tb in tb_dict.items():
        assert tb.id == tb_id

        # iterate over sentences
        sent_index = None
        token_start_index = None
        token_end_index = None
        for i, sent_offsets in enumerate(offsets):

            sent_start = sent_offsets[0][0]
            sent_end = sent_offsets[-1][-1]

            sent_start_match = start_match(tb.start, sent_start, sent_end)
            sent_end_match =   end_match(tb.end,     sent_start, sent_end)

            # text bound in sentence
            if sent_start_match:
                sent_index = i

                if not sent_end_match:
                    logging.warn(f"Textbound end not in same sentences start: {tb}")

                # iterate over tokens
                for j, (token_start, token_end) in enumerate(sent_offsets):
                    if start_match(tb.start, token_start, token_end):
                        token_start_index = j
                    if end_match(tb.end, token_start, token_end):
                        token_end_index = j + 1
                break

        assert sent_index is not None
        assert token_start_index is not None
        if token_end_index is None: #assign the token index of the last token in the sentence as the end of the TB
            logging.warn(f"Token end index is None")
            token_end_index = len(offsets[sent_index])

        map[tb_id] = (sent_index, token_start_index, token_end_index)

    return map


def rm_ws(spacy_tokens):
    return [token for token in spacy_tokens if token.text.strip()]

def tokenize_document(text, tokenizer):

    doc = tokenizer(text)

    #sent_bounds = []
    tokens = []
    offsets = []
    for sent in rm_ws(doc.sents):
        #sent_bounds.append((sent.start_char, sent.end_char))
        sent = rm_ws(sent)

        tok = [t.text for t in sent]
        os = [(t.idx, t.idx + len(t.text)) for t in sent]

        tokens.append(tok)
        offsets.append(os)

    # Check
    for tok, off in zip(tokens, offsets):
        for t, o in zip(tok, off):
            assert t == text[o[0]:o[1]]

    #return (sent_bounds, tokens, offsets)
    return (tokens, offsets)


def convert_doc(text, ann, id, tokenizer, \
                    allowable_tb = None,
                    relation_default = RELATION_DEFAULT):


    tokens, offsets = tokenize_document(text, tokenizer)


    # Extract events, text bounds, and attributes from annotation string
    event_dict, relation_dict, tb_dict, attr_dict = get_annotations(ann)

    # summary = get_summary(event_dict, relation_dict, attr_dict)

    indices = get_tb_indices(tb_dict, offsets)

    sent_count = len(tokens)

    entities = [OrderedDict() for _ in range(sent_count)]
    subtypes = [OrderedDict() for _ in range(sent_count)]
    relations = [[] for _ in range(sent_count)]

    if event_dict != {}:
        for event_id, event in event_dict.items():
            #print()
            #print(event_id, event)

            head_tb_id = None
            head_sent = None
            head_index = None

            for i, (tb_type, tb_id) in enumerate(event.arguments.items()):
                #print()

                sent_index, token_start, token_end = indices[tb_id]
                #print("sentence index", sent_index, token_start, token_end)
                #print(tb_type, tb_id)
                tb = tb_dict[tb_id]
                #print(tb)


                if tb_id in attr_dict:
                    attr_type = attr_dict[tb_id].type_
                    attr_value = attr_dict[tb_id].value
                else:
                    attr_type = tb.type_
                    attr_value = tb.type_

                if tb.type_ not in attr_type:
                    logging.warn(f"Attribute type not in textbound type: {tb.type_} not in {attr_type}")


                #print(attr)
                if (allowable_tb is None) or (tb.type_ in allowable_tb):

                    if tb_id not in entities[sent_index]:
                        d = {SPERT_TYPE: tb.type_, SPERT_START: token_start, SPERT_END: token_end}
                        entities[sent_index][tb_id] = d

                        d = {SPERT_TYPE: attr_value, SPERT_START: token_start, SPERT_END: token_end}
                        subtypes[sent_index][tb_id] = d

                    entity_index = list(entities[sent_index].keys()).index(tb_id)


                    if i == 0:
                        head_tb_id = tb_id
                        head_sent = sent_index
                        head_index = entity_index
                    elif head_sent == sent_index:

                        assert head_tb_id is not None
                        assert head_sent is not None
                        assert head_index is not None

                        d = {SPERT_TYPE: relation_default, SPERT_HEAD: head_index, SPERT_TAIL: entity_index}
                        relations[sent_index].append(d)

                    else:
                        logging.warn(f"Head index not an same sentence as tail. Skipping relation.")

    if tb_dict != {} and event_dict == {}:
        for tbid in indices:
            sent_index,token_start,token_end = indices[tbid]
            tb = tb_dict[tbid]
            d = {SPERT_TYPE: tb.type_, SPERT_START: token_start, SPERT_END: token_end}
            entities[sent_index][tbid] = d

        
    #print(entities)
    entities = [list(sent.values()) for sent in entities]
    subtypes = [list(sent.values()) for sent in subtypes]
    #print(entities)

    #print(relations)

    assert len(tokens) == sent_count
    assert len(entities) == sent_count
    assert len(relations) == sent_count

    out = []
    for i in range(sent_count):
        d = {}
        d[SPERT_ID] = f'{id}[{i}]'
        d[SPERT_TOKENS] = tokens[i]
        d[SPERT_OFFSETS] = offsets[i]
        d[SPERT_ENTITIES] = entities[i]
        d[SPERT_SUBTYPES] = subtypes[i]
        d[SPERT_RELATIONS] = relations[i]
        
        out.append(d)

    return out

def get_allowable_types(path):

    if path is None:
        return None
    else:
        types = json.load(open(path, 'r'))
        #d = {}
        #d['entities'] = list(types['entities'].keys())
        #d['relations'] = list(types['relations'].keys())

        allowable_tb = list(types['entities'].keys())
        return allowable_tb



