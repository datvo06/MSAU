import os

def read_image_list(pathToList, prefix=None):
    '''

    :param pathToList:
    :return:
    '''
    f = open(pathToList, 'r')
    filenames = []
    for line in f:
        if line[0] != '#':
            if line[-1] == '\n':
                filename = line[:-1]
            else:
                filename = line
            if prefix is not None:
                filename = prefix + filename
            filenames.append(filename)
        else:
            if len(line) < 3:
                break
    f.close()
    return filenames
