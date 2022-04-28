import os
import pickle
from glob import glob


def count_token(dir_path, genre=None, logscale=False):
    if genre == 'classic':
        files = glob(os.path.join(dir_path, '*/*'))
    else:
        files = glob(os.path.join(dir_path, '*'))

    count_list = []
    sequence_len = []

    noteon_list = []
    velocity_list = []
    duration_list = []
    noteoff_list = []
    timeshift_list =[]

    for file_path in files:
        file = os.path.join(dir_path, file_path)

        with open(file, "rb") as f:
            data = pickle.load(f)

        if not logscale:
            noteon_list.append(len([i for i in data if i in range(0, 128)]))
            noteoff_list.append(len([i for i in data if i in range(128, 256)]))
            timeshift_list.append(len([i for i in data if i in range(256, 356)]))
            velocity_list.append(len([i for i in data if i in range(356, 388)]))
        else:
            noteon_list.append(len([i for i in data if i in range(0, 128)]))
            duration_list.append(len([i for i in data if i in range(128, 158)]))
            timeshift_list.append(len([i for i in data if i in range(158, 188)]))
            velocity_list.append(len([i for i in data if i in range(188, 220)]))

        sequence_len.append(len(data))
        temp = set(data)

        count_list.append(len(temp))

    print("mean sequence", sum(sequence_len)/len(sequence_len))

    print("noteon sequence", sum(noteon_list)/len(noteon_list))
    if not logscale:
        print("noteoff sequence", sum(noteoff_list)/len(noteoff_list))
    else:
        print("duration sequence", sum(duration_list)/len(duration_list))
    print("velocity sequence", sum(velocity_list)/len(velocity_list))
    print("timeshift sequence", sum(timeshift_list)/len(timeshift_list))

    return sum(count_list)/len(count_list)


print("평균적으로 사용하는 토큰 종류 크기")

print("MIDI-like Classic :", count_token('/home/bang/PycharmProjects/MusicGeneration/MusicTransformer-Pytorch/dataset/e_piano', 'classic'))
print("logscale Classic :", count_token('/home/bang/PycharmProjects/MusicGeneration/MusicTransformer-Pytorch/dataset/logscale_epiano0420', 'classic', logscale=True))

print()

print("MIDI-like POP :", count_token('/home/bang/PycharmProjects/MusicGeneration/MusicTransformer-Pytorch/dataset/pop_pickle', 'pop'))
print("logscale POP :", count_token('/home/bang/PycharmProjects/MusicGeneration/MusicTransformer-Pytorch/dataset/logscale_pop0420', 'pop', logscale=True))