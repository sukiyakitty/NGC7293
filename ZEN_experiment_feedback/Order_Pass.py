import os
import time
import tkinter as tk
from tkinter import messagebox


class verification_window(tk.Frame):

    # 调用时初始化
    def __init__(self):
        global root, wrong_time
        wrong_time = 3
        root = tk.Tk()

        screen_width = root.winfo_screenwidth()  # 得到屏幕宽度
        screen_height = root.winfo_screenheight()  # 得到屏幕高度
        windows_width = 400  # 窗口大小
        windows_height = 300
        x = int((screen_width - windows_width) / 2)
        y = int((screen_height - windows_height) / 2)
        root.geometry("%dx%d+%d+%d" % (windows_width, windows_height, x, y))
        root.resizable(0, 0)  # 窗口大小固定
        root.title('Password!')

        super().__init__()
        self.username = tk.StringVar()
        self.password = tk.StringVar()
        self.pack()
        self.main_window()

        root.mainloop()

    # 窗口布局
    def main_window(self):
        global root, wrong_time
        username_label = tk.Label(root, text='Username:', font=('Arial', 24)).place(x=50, y=10)
        username_input = tk.StringVar
        username_entry = tk.Entry(root, textvariable=self.username, font=('Arial', 24)).place(x=20, y=65)
        password_label = tk.Label(root, text='Password:', font=('Arial', 24)).place(x=50, y=120)
        password_input = tk.StringVar
        password_entry = tk.Entry(root, textvariable=self.password, show='*', font=('Arial', 24)).place(x=20, y=175)

        # 在按下CONFIRM按钮时调用验证函数
        conformation_button = tk.Button(root, text='CONFIRM', command=self.verification, fg='white', bg='black',
                                        activeforeground='white', activebackground='navy', width=24, height=2)
        conformation_button.place(x=20, y=250)
        quit_button = tk.Button(root, text='QUIT', command=root.quit, fg='white', bg='black', activeforeground='white',
                                activebackground='red', width=24, height=2)
        quit_button.place(x=220, y=250)

    # 验证函数
    def verification(self):
        global root, wrong_time

        # 检查用户名和密码 是否在user_dict字典中
        user_dict = {'kitty': 'jou8JJK7', 'YXC': 'cellFATE', 'SQS': '18811445859'}
        if user_dict.get(str(self.username.get())) == str(self.password.get()):
            # 成功提醒
            messagebox.showinfo(title='Correct',
                                message=f'{str(self.username.get())}, welcome!\nSuccess Pause Experiment!\nPlease wait until this end of Experiment loop ...\nThe Experiment will Pause automatically!')
            self.do()
            root.quit()
            # 验证成功后打开main_gui窗口
            # root.withdraw()
            # from gui import main_gui
            # main_gui.app()
        else:
            wrong_time -= 1
            messagebox.showerror(title='Wrong password!',
                                 message='Wrong password!\nPlease enter correct username or password.\nPause Experiment will be ignored!')  # 错误提醒
            if wrong_time <= 0:
                root.quit()

    def do(self):
        main_path = r'C:\Py\Magic.cdn'
        if not os.path.exists(main_path):
            print('没有实验进行！\n或程序内部错误，该程序遇到问题需要关闭！\n请联系系统开发人员！')
            messagebox.showerror(title='程序内部错误!',
                                 message='没有实验进行！\n程序内部错误，该程序遇到问题需要关闭！\n请联系系统开发人员！')
            root.quit()
            return False
        for i in range(1):
            magic_file = open(main_path, 'r+')
            magic_file.truncate()
            magic_file.seek(0)
            magic_file.write('PauseExperiment()')
            magic_file.close()
            time.sleep(0.21)


if __name__ == '__main__':
    verification_window()
