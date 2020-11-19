import time


def main():
    print('Now I will try to Pause the Experiment...')
    print('Please wait...')
    for j in range(21):
        print('.', end='')
        time.sleep(0.13)
    print('.')
    for i in range(1):
        magic_file = open('C:\\Py\\Magic.cdn', 'r+')
        magic_file.truncate()
        magic_file.seek(0)
        magic_file.write('PauseExperiment()')
        magic_file.close()
        print('Success! please wait until this end of Experiment loop ...')
        time.sleep(0.21)
    print('Bye! ...')
    time.sleep(6)


if __name__ == '__main__':
    main()
