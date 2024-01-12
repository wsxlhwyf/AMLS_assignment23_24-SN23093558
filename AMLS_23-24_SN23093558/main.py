import os

if __name__ == "__main__":
    for i in range(3):
        user_input = input("do u want to run A or B ?")
        if user_input == 'A':
            os.system('python ./A/A.py')
        elif user_input == 'B':
            os.system('python ./B/B.py')
        else:
            print('wrong mode!')