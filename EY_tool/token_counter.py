import os
import pickle
from glob import glob


def count_token(dir_path, genre=None):
    if genre == 'classic':
        files = glob(os.path.join(dir_path, '*/*'))
    else:
        files = glob(os.path.join(dir_path, '*'))

    count_list = []
    sequence_len = []

    for file_path in files:
        file = os.path.join(dir_path, file_path)

        with open(file, "rb") as f:
            data = pickle.load(f)

        sequence_len.append(len(data))
        temp = set(data)

        count_list.append(len(temp))

    print(sum(sequence_len)/len(sequence_len))
    return sum(count_list)/len(count_list)


print("평균적으로 사용하는 토큰 종류 크기")

print("MIDI-like Classic :", count_token('/home/bang/PycharmProjects/MusicGeneration/MusicTransformer-Pytorch/dataset/e_piano', 'classic'))
print("relative Classic :", count_token('/home/bang/PycharmProjects/MusicGeneration/MusicTransformer-Pytorch/dataset/octave_fusion_absolute_e_piano', 'classic'))

print()

print("MIDI-like POP :", count_token('/home/bang/PycharmProjects/MusicGeneration/MusicTransformer-Pytorch/dataset/pop_pickle', 'pop'))
print("relative POP :", count_token('/home/bang/PycharmProjects/MusicGeneration/MusicTransformer-Pytorch/dataset/pop909_absolute', 'pop'))