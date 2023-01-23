import os
TEXTBOUND_LB_SEP = ';'
import constants as C


class Attribute(object):
    '''
    Container for attribute

    annotation file examples:
        A1      Value T2 current
        A2      Value T8 current
        A3      Value T9 none
        A4      Value T13 current
        A5      Value T17 current
    '''
    def __init__(self, id, type_, textbound, value):
        self.id = id
        self.type_ = type_
        self.textbound = textbound
        self.value = value

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return (self.type_ == other.type_) and \
               (self.textbound == other.textbound) and \
               (self.value == other.value)
    def brat_str(self):
        return attr_str(attr_id=self.id, arg_type=self.type_, \
                            tb_id=self.textbound, value=self.value)

    def id_numerical(self):
        assert self.id[0] == 'A'
        id = int(self.id[1:])
        return id


class Textbound(object):
    '''
    Container for textbound

    Annotation file examples:
        T2	Tobacco 30 36	smoker
        T4	Status 38 46	Does not
        T5	Alcohol 47 62	consume alcohol
        T6	Status 64 74	No history
    '''
    def __init__(self, id, type_, start, end, text, tokens=None):
        self.id = id
        self.type_ = type_
        self.start = start
        self.end = end
        self.text = text
        self.tokens = tokens

    def __str__(self):
        return str(self.__dict__)

    # def token_indices(self, char_indices):
    #     i_sent, (out_start, out_stop) = find_span(char_indices, self.start, self.end)
    #     return (i_sent, (out_start, out_stop))

    def brat_str(self):
        return textbound_str(id=self.id, type_=self.type_, start=self.start, \
                                                end=self.end, text=self.text)

    def id_numerical(self):
        assert self.id[0] == 'T'
        id = int(self.id[1:])
        return id


class Event(object):
    '''
    Container for event

    Annotation file examples:
        E3      Family:T7 Amount:T8 Type:T9
        E4      Tobacco:T11 State:T10
        E2      Alcohol:T5 State:T4

        id     event:head (entities)
    '''

    def __init__(self, id, type_, arguments):
        self.id = id
        self.type_ = type_
        self.arguments = arguments

    def get_trigger(self):
        for argument, tb in self.arguments.items():
            return (argument, tb)

    def __str__(self):
        return str(self.__dict__)

    def brat_str(self):
        return event_str(id=self.id, event_type=self.type_, \
                            textbounds=self.arguments)


class Relation(object):
    '''
    Container for event

    Annotation file examples:
    R1  attr Arg1:T2 Arg2:T1
    R2  attr Arg1:T5 Arg2:T6
    R3  attr Arg1:T7 Arg2:T1

    '''

    def __init__(self, id, role, arg1, arg2):
        self.id = id
        self.role = role
        self.arg1 = arg1
        self.arg2 = arg2

    def __str__(self):
        return str(self.__dict__)

    def brat_str(self):
        return relation_str(id=self.id, role=self.role, \
                            arg1=self.arg1, arg2=self.arg2)



def textbound_str(id, type_, start, end, text):
    '''
    Create textbounds during from span

    Parameters
    ----------
    id: current textbound id as string
    span: Span object

    Returns
    -------
    BRAT representation of text bound as string
    '''

    if '\n' in text:
        i = 0
        substrings = text.split('\n')
        indices = []
        for s in substrings:
            n = len(s)
            idx = '{start} {end}'.format(start=start + i, end=start + i + n)
            indices.append(idx)
            i += n + 1

        indices = TEXTBOUND_LB_SEP.join(indices)

    else:
        indices = '{start} {end}'.format(start=start, end=end)

    text = re.sub('\n', ' ', text)


    if isinstance(id, str) and (id[0] == "T"):
        id = id[1:]

    return 'T{id}\t{type_} {indices}\t{text}'.format( \
        id = id,
        type_ = type_,
        indices = indices,
        text = text)

    #return 'T{id}\t{type_} {start} {end}\t{text}'.format( \
    #    id = id,
    #    type_ = type_,
    #    start = start,
    #    end = end,
    #    text = text)


def attr_str(attr_id, arg_type, tb_id, value):
    '''
    Create attribute string
    '''

    if isinstance(attr_id, str) and (attr_id[0] == "A"):
        attr_id = attr_id[1:]

    if isinstance(tb_id, str) and (tb_id[0] == "T"):
        tb_id = tb_id[1:]

    return 'A{attr_id}\t{arg_type} T{tb_id} {value}'.format( \
        attr_id = attr_id,
        arg_type = arg_type,
        tb_id = tb_id,
        value = value)


def event_str(id, event_type, textbounds):
    '''
    Create event string

    Parameters:
    -----------
    id: current event ID as string
    event_type: event type as string
    textbounds: list of tuple, [(span.type_, id), ...]

    '''

    if isinstance(id, str) and (id[0] == "E"):
        id = id[1:]

    # Start event string
    out = 'E{}\t'.format(id)

    # Create event representation
    event_args = []
    for arg_type, tb_id in textbounds.items():

        if tb_id[0] == "T":
            tb_id = tb_id[1:]

        if arg_type == C.TRIGGER:
            arg_type = event_type

        out += '{}:T{} '.format(arg_type, tb_id)

    return out


def relation_str(id, role, arg1, arg2):
    '''
    Create event string

    Parameters:
    -----------

    R1	attr Arg1:T2 Arg2:T1

    '''


    if isinstance(id, str) and (id[0] == "R"):
        id = id[1:]

    # Start event string
    out = f'R{id}\t{role} Arg1:{arg1} Arg2:{arg2}'

    return out




def write_file(path, id, content, ext):

    # Output file name
    fn = os.path.join(path, '{}.{}'.format(id, ext))

    # Directory, including path in id
    dir_ = os.path.dirname(fn)
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    # Write file
    with open(fn, 'w', encoding=C.ENCODING) as f:
        f.write(content)


    #os.chmod(dir_, stat.S_IWGRP)
    #os.chmod(fn, stat.S_IWGRP)
    return fn



def write_txt(path, id, text, fix_linebreak=False, strip_ws=False):
    '''
    Write text file
    '''

    fix_flag=0
    if fix_linebreak:
        max_len = 0
        sentence_list = text.split('\n')

        for sentence in sentence_list:
            if len(sentence)>max_len:
                max_len=len(sentence)

        # Consider only reports where the maximum char length for each sentence is shorter than 100
        if max_len<=100:
            fix_flag=1

    if fix_flag==1:
        # Find indices where the extraneous linebreak should be fixed
        updated_indices = []
        for match in re.finditer(r'[A-Za-z,] *\n *[A-Za-z0-9]', text):
            index = match.start()+match.group().find('\n')
            updated_indices.append(index)

            # updated_text = re.sub(r'[A-Za-z,] *\n *[A-Za-z0-9]', convert, text)

        # Among found indices, remove those who are followed by ':' in the next sentence
        updated_indices_2 = []
        for updated_index in updated_indices:
            if ':' in text[updated_index:].split('\n')[1]:
                pass
            else:
                updated_indices_2.append(updated_index)

        updated_text = ''
        if len(updated_indices_2)==0:
            updated_text = text
        else:
            for char_idx in range(len(text)):
                if char_idx in updated_indices_2:
                    updated_text+=' '
                else:
                    updated_text+=text[char_idx]

        text = updated_text

    if strip_ws:
        text = '\n'.join([line.strip() for line in text.splitlines()])

    fn = write_file(path, id, text, C.TEXT_FILE_EXT)
    return fn

def write_ann(path, id, ann):
    '''
    Write annotation file
    '''
    fn = write_file(path, id, ann, C.ANN_FILE_EXT)
    return fn

def get_max_id(object_dict):
    x = [v.id_numerical() for k, v in object_dict.items()]
    if len(x) == 0:
        i = 0
    else:
        i = max(x)
    return i


def get_next_index(d):
    ids = [x.id for x in d.values()]

    if len(ids) == 0:
        last = 0
    else:
        ids = [int(id[1:]) for id in ids]
        last = max(ids)

    return last + 1