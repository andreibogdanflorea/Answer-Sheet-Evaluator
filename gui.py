import test_evaluator as testeval
import tkinter as tk
import tkinter.filedialog
import pandas as pd
from tkinter import messagebox

# model_development.create_train_save_letter_model("models/best_trained_model")

root = tk.Tk()
root.title('Test Evaluator')
root.resizable(0, 0)
bg_color = '#253E53'
canvas = tk.Canvas(root, height=600, width=1100, bg=bg_color)
canvas.pack()

frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.9, relheight=0.9, relx=0.05, rely=0.05)


def quit_protocol():
    root.quit()


def save_to_excel(results):
    file_types = [('All Files', '*.*'), ('Excel Files', '*.xls')]
    path = tk.filedialog.asksaveasfilename(initialdir='.', title='Save results to Excel file',
                                           filetypes=file_types, defaultextension=file_types)
    if not (str(path).endswith(".xls") or str(path).endswith(".xlsx")):
        messagebox.showerror(title='Error', message='Extension should be .xls or .xlsx')
        return

    df = pd.DataFrame(results, columns=['First Name', 'Last Name', 'Score'])
    df.to_excel(path)
    savedatalabel = tk.Label(frame, text='Saved file.', padx=10, pady=5,
                             fg='black', bg='white', width=20)
    savedatalabel.place(relx=0.08, rely=0.65)


def openfiledialog():
    images_paths = tk.filedialog.askopenfilenames(initialdir=".", title='Select scanned Answer Sheets',
                                                  filetypes=(('PNG Images', '*.png'), ('JPG Images', '*.jpg')))

    results = []
    for image_path in images_paths:
        first_name, last_name, ans_results = testeval.get_name_and_answers(image_path)
        score = testeval.compute_score(ans_results)
        print('''Participant {} {} had answers {}, totalling a score of {}.'''
              .format(first_name.capitalize(), last_name.capitalize(), ans_results, score))
        results.append((first_name, last_name, score))

    rows = len(results)
    columns = 3

    resultslabel = tk.Label(frame, text='Results', padx=10, pady=5,
                            fg='black', bg='white', width=20)
    resultslabel.place(relx=0.635, rely=0.07)

    tableframe = tk.Frame(frame, bg='white')
    tableframe.place(relwidth=0.5, relheight=0.9, relx=0.5, rely=0.15)

    header = ['First Name', 'Last Name', 'Score']
    for j in range(len(header)):
        e = tk.Entry(tableframe, width=20, fg='black', bg='white')
        e.grid(row=0, column=j)
        e.insert(tk.END, header[j])

    for i in range(rows):
        for j in range(columns):
            e = tk.Entry(tableframe, width=20, fg='black', bg='white')
            e.grid(row=i + 1, column=j)
            e.insert(tk.END, results[i][j])

    savedata = tk.Button(frame, text='Save as Excel file', padx=10, pady=5, fg='white', bg=bg_color,
                         command=lambda: save_to_excel(results), width=19)
    savedata.place(relx=0.08, rely=0.6)


openFile = tk.Button(frame, text='Open File', padx=10, pady=5, fg='white', bg=bg_color,
                     command=openfiledialog, width=19)
openFile.place(relx=0.08, rely=0.2)

root.protocol("WM_DELETE_WINDOW", quit_protocol())
root.mainloop()
