"""
For python version > 3
"""

import src.utils.Folders as folders
import src.config.GlobalParameters as GP
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext


class MouseOverButton(tk.Button):
    def __init__(self, master, **kw):
        tk.Button.__init__(self, master=master, **kw)
        self.defaultBackground = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self['background'] = self['activebackground']

    def on_leave(self, e):
        self['background'] = self.defaultBackground

class ConsciencesFWIU():
    """
    A class that can start an IU process to write, from buttons, words into python console.
    """
    __owner = "ConsciencesAI"
    __creation_date = "28/09/2019"

    def __init__(self, resolution, full_options=True, load_configuration=False):
        self.__check_resolution(resolution)
        self.full_options = full_options
        self.load_configuration = load_configuration

    def __check_resolution(self, resolution):
        """
        Check if resolution is a list or tuple of at least 2 elements.
        Args:
            resolution: A list or tuple.

        Returns:

        """
        if isinstance(resolution, (list, tuple)):
            if len(resolution) > 1:
                self.resolution = resolution
            else:
                raise ValueError("'resolution' len() has to be > 1.")
        else:
            raise TypeError("'resolution' is not type() list or tuple.")

    def start(self):
        """
        Create a IU process to write, from buttons, words into python console.

        Returns:

        """
        self.root = self.__root_window()

        tkinterwidget = TkinterWidges(cls=self)
        tkinterwidget.top_frame()
        self.code_label_frame = tkinterwidget.middle_frame()
        self.options_frame, self.text_code_frame, self.wait_code_frame = tkinterwidget.frames_in_middle_frame()
        self.options_button, self.input_code, self.txt_wait = tkinterwidget.middle_frame_buttons()
        self.frame_buttons = tkinterwidget.botton_frame()
        tkinterwidget.botton_frame_buttons()
        self.root.mainloop()

    def __root_window(self):
        """
        Create the main (root) window of the UI.

        Returns: The root window.

        """
        root = tk.Tk()
        root.title("Consciences IU")
        root.minsize(270, 333)
        x, y, _, _ = TkinterUtils.center_window(root, self.resolution[0], self.resolution[1])
        root.geometry("x".join(map(str, self.resolution)) + "+" + str(x) + "+" + str(y))
        root.iconbitmap(GP.icon_ico)
        root.config(bg="#f5f2f2")

        root.rowconfigure(1, weight=1)  # To expand row inside main window.
        root.columnconfigure(0, weight=1)  # To keep widgets in place.

        return root

    def _help_button(self):
        self.input_code.insert(tk.END, "\nHELP")

    def _toplevel_window_plus_button(self):
        """
        Create a button that bring up a new window with an Entry widget to retrieve "Project's name" an two
        buttons, "Create" and "Cancel".
        Returns:

        """
        # Create the toplevel window.
        self.top = tk.Toplevel(self.root)
        self.top.title("Options")
        self.top.resizable(False, False)
        self.top.iconbitmap(GP.icon_ico)
        self.top.config(bg="#ffffff")

        # Define the interactive widgets inside toplevel window in plus button.
        project_name = tk.Label(self.top, text="Project's name:", font=("Segoe UI", "8", "bold"), bg="#ffffff")
        project_name.grid(row=0, column=0, pady=10, padx=10)
        input_project = tk.Entry(self.top, bg="#f5f2f2", relief="sunken")
        input_project.grid(row=0, column=1, pady=10, padx=10)
        input_project.focus()
        photo_consciences = tk.PhotoImage(file=GP.help_png)
        image_consciences = tk.Label(self.top, image=photo_consciences, bg="#ffffff")
        image_consciences.image = photo_consciences  # To keep a reference to prevent from getting it
        # garbage-collected.
        image_consciences.grid(row=0, column=2, columnspan=2, sticky=tk.W + tk.E + tk.N + tk.S)
        create_button = MouseOverButton(self.top, text="Create", font=("Segoe UI", "8"), bg="#e3e1e1",
                                        command=self._create_button)
        create_button.grid(row=1, column=2, sticky=tk.W + tk.E + tk.N + tk.S)
        cancel_button = MouseOverButton(self.top, text="Cancel", font=("Segoe UI", "8"), bg="#e3e1e1",
                                        command=self._exit_top)
        cancel_button.grid(row=1, column=3, sticky=tk.W + tk.E + tk.N + tk.S)

        # To center the toplevel window in the screen.
        x, y, width, height = TkinterUtils.center_window(self.top, update=True)
        self.top.geometry('%dx%d+%d+%d' % (width, height, x, y))

        # To disable the options button while the toplevel window is opem.
        self.options_button.config(state="disable")

        self.top.protocol("WM_DELETE_WINDOW", self._exit_top)

    # root.
    def _exit_top(self):
        self.top.destroy()
        self.options_button.config(state='normal')

    def _send_button(self):
        self.input_code.insert(tk.END, "\nSENT")

    def _wait_button(self):
        if len(self.txt_wait.get()) == 0:
            pass
        else:
            self.input_code.insert(tk.END, "\nWAIT - " + self.txt_wait.get())

    def _save_button(self):
        self.input_code.insert(tk.END, "\nSAVE")

    def _stop_button(self):
        self.input_code.insert(tk.END, "\nSTOP")

    def _create_button(self):
        folders.copy_entire_directory_to_path(path_to_be_copied=GP.DEFAULT_PROJECT_ID_PATH,
                                              path_to_be_paste=GP.PROJECTS_PATH + "new\\")
        self.input_code.insert(tk.END, "\nCOPIED!")
        self._exit_top()

class TkinterWidges():

    def __init__(self, cls: ConsciencesFWIU):
        self.cls = cls

    def top_frame(self):
        """
        Create the top frame inside the root window, where the help button is placed.

        Returns:

        """
        frame_top = tk.Frame(self.cls.root, bg="#f5f2f2")
        frame_top.grid(sticky=tk.W + tk.E)
        frame_inside_top = tk.Frame(frame_top)  # A blank frame.
        frame_inside_top.grid(row=0, column=0)

        photo_help = tk.PhotoImage(file=GP.help_png)
        btn_help = MouseOverButton(frame_top, image=photo_help, bg="#e3e1e1", command=self.cls._help_button)
        btn_help.grid(column=1, padx=2, pady=4)
        btn_help.image = photo_help  # To keep a reference to prevent from getting it garbage-collected.

        frame_top.columnconfigure(0, weight=1)

    def middle_frame(self):
        """
        Create the white middle frame inside the root window. Then create a Label frame inside the white frame,
        with 2 blank frames to keep proportions.

        Returns: Return the Label frame where the different widgets are going to be placed.

        """
        white_frame_middle = tk.Frame(self.cls.root, bg="#ffffff", bd=2, relief="groove")  # White middle frame.
        white_frame_middle.grid(padx=10, sticky=tk.W + tk.E + tk.N + tk.S)
        code_label_frame = tk.LabelFrame(white_frame_middle, font=("Segoe UI", "9", "bold"), text="Code Input",
                                         bg="#ffffff", bd=1)
        code_label_frame.grid(row=0, padx=20, pady=20, sticky=tk.W + tk.E + tk.N + tk.S)
        lat_blank_frame_in_white_frame = tk.Frame(white_frame_middle,
                                                  bg="#ffffff")  # Lateral blank frame inside white middle frame.
        lat_blank_frame_in_white_frame.grid(row=0, column=1, ipadx=20)
        bot_blank_frame_in_white_frame = tk.Frame(white_frame_middle,
                                                  bg="#ffffff")  # Bottom blank frame inside white middle frame.
        bot_blank_frame_in_white_frame.grid(row=1, ipady=20)

        # Columns and rows configuration
        white_frame_middle.columnconfigure(0, weight=1)  # To expand column inside white middle frame.
        white_frame_middle.rowconfigure(0, weight=1)  # To expand row inside white middle frame.
        code_label_frame.columnconfigure(0, weight=1)  # To expand column inside LabelFrame.
        code_label_frame.rowconfigure(1, weight=1)  # To expand column inside LabelFrame.

        return code_label_frame

    def frames_in_middle_frame(self):
        """
        Create the frames to sort the interactive widgets inside the middle frame. Add a new frame here to sort
        each group of interactive widgets.

        Returns: options_frame, text_code_frame, wait_code_frame frames where the interactive widgets are placed.

        """
        options_frame = tk.Frame(self.cls.code_label_frame, bg="#ffffff")
        options_frame.grid()
        text_code_frame = tk.Frame(self.cls.code_label_frame, bg="#ffffff")  # Text code frame.
        text_code_frame.grid(sticky=tk.W + tk.E + tk.N + tk.S)
        wait_code_frame = tk.Frame(self.cls.code_label_frame, bg="#ffffff")  # Wait button frame.
        wait_code_frame.grid(sticky=tk.W)

        # Columns and rows configuration
        options_frame.rowconfigure(0, weight=1)  # To expand row inside input code frame.
        options_frame.columnconfigure(0, weight=1)  # To expand column inside input code frame.
        text_code_frame.columnconfigure(0, weight=2)  # To expand column inside input code frame.
        text_code_frame.rowconfigure(0, weight=1)  # To expand row inside input code frame.
        wait_code_frame.columnconfigure(0, weight=1)  # To keep column aspect where "Waiting time:" frame is placed.
        wait_code_frame.columnconfigure(1, weight=1)  # To keep column aspect where Entry frame is placed.
        wait_code_frame.columnconfigure(2, weight=1)  # To keep column aspect where Wait button frame is placed.

        return options_frame, text_code_frame, wait_code_frame

    def middle_frame_buttons(self):
        """
        Define the widgets placed in the frames inside the middle frame.

        Returns: options_button, the plus button in options_frame.
                 input_code, the ScrolledText to receive the input.
                 txt_wait, the Entry widget in the wait_code_frame.

        """
        options_button = TkinterWidges.options_(self)
        input_code = TkinterWidges.text_(self)
        txt_wait = TkinterWidges.wait_(self)

        return options_button, input_code, txt_wait

    def options_(self):
        """
        Define widgets in the options frame.

        Returns:

        """

        options_Spinbox = ttk.Combobox(self.cls.options_frame, state="readonly")
        options_Spinbox.grid(row=0, pady=10, padx=10)

        # # Synchronise the root window movement with the toplevel window.
        # def sync_windows(event=None):
        #     x = root.winfo_x() + root.winfo_width() + 4
        #     y = root.winfo_y()
        #     top.geometry("+%d+%d" % (x, y))
        #
        # root.bind("<Configure>", sync_windows)

        options_button = MouseOverButton(self.cls.options_frame, font=("Segoe UI", "8"), text="+", bg="#e3e1e1",
                                         command=self.cls._toplevel_window_plus_button)
        options_button.grid(row=0, column=1, padx=10)

        return options_button

    def text_(self):
        """
        Define widgets in the text code frame.

        Returns: input_code, the ScrolledText widget.

        """
        input_code = scrolledtext.ScrolledText(self.cls.text_code_frame, bg="#f5f2f2")
        input_code.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W + tk.E + tk.N + tk.S)
        input_code.focus()

        btn_code_send = MouseOverButton(self.cls.text_code_frame, font=("Segoe UI", "8"), text="Send", bg="#e3e1e1",
                                        command=self.cls._send_button)
        btn_code_send.grid(row=0, column=1, padx=10)

        return input_code

    def wait_(self):
        """
        Define widgets in the wait code frame.

        Returns: txt_wait, the Entry widget in the wait_code_frame.

        """
        text_wait = tk.Label(self.cls.wait_code_frame, font=("Segoe UI", "8"), text="Waiting time:", bg="#ffffff")
        text_wait.grid(row=0, column=0, padx=10, pady=30)

        txt_wait = tk.Entry(self.cls.wait_code_frame, bg="#f5f2f2", relief="sunken", width=4)
        txt_wait.grid(row=0, column=1, sticky=tk.W)

        btn_wait = MouseOverButton(self.cls.wait_code_frame, font=("Segoe UI", "8"), text="Wait", bg="#e3e1e1",
                                   command=self.cls._wait_button)
        btn_wait.grid(row=0, column=2, padx=10)

        return txt_wait

    def botton_frame(self):
        """
        Create the botton frame where the "Save" and "Stop" butons are placed.

        Returns: Return the botton frame.

        """
        frame_buttons = tk.Frame(self.cls.root, bg="#f5f2f2")
        frame_buttons.grid(sticky=tk.W + tk.E)
        first_blank_column = tk.Frame(frame_buttons, bg="#f5f2f2") # A blank frame.
        first_blank_column.grid(row=0, column=0)

        frame_buttons.columnconfigure(0, weight=1)  # To extend blank frame.

        return frame_buttons

    def botton_frame_buttons(self):
        """
        Define the buttons inside the botton_frame.

        Returns:

        """
        btn_save = MouseOverButton(self.cls.frame_buttons, font=("Segoe UI", "8"), text="Save", width=7, bg="#e3e1e1",
                                   command=self.cls._save_button)
        btn_save.grid(row=0, column=1, padx=1, pady=8)
        btn_stop = MouseOverButton(self.cls.frame_buttons, font=("Segoe UI", "8"), text="Stop", width=7, bg="#e3e1e1",
                                   command=self.cls._stop_button)
        btn_stop.grid(row=0, column=2, padx=10, pady=8)

class TkinterUtils():

    @staticmethod
    def center_window(win, width=None, height=None, update=False):
        """
        Make the window to be centered on the screen when it pop-up.
        Args:
            win: Window to be centered.
            width: Window width.
            height: Window heigth.
            update: In case the window resolution can't be retrieved, set this to True.

        Returns: Window position and window resolution.

        """
        if update == True:
            win.update_idletasks()
            width = win.winfo_reqwidth()
            height = win.winfo_reqheight()

        screen_width = win.winfo_screenwidth()  # Get the screen resolution
        screen_height = win.winfo_screenheight()

        x = screen_width / 2 - width / 2
        y = screen_height / 2 - height / 2

        return int(x), int(y), int(width), int(height)

if __name__=="__main__":
    consci = ConsciencesFWIU((500, 500))
    consci.start()
