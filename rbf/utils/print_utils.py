"""
print_utils.py

Utility functions for printing fancy messages.
"""

import textwrap

class color:
    purple = '\033[95m'
    cyan = '\033[96m'
    darkcyan = '\033[36m'
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    bold = '\033[1m'
    end = '\033[0m'

def print_with_box(text: str, box_color: str = color.purple, text_color: str = color.end, title: str = "", max_len = 88) -> None:
    """
    Prints a message with a box around it.
    """
    lines = text.split("\n")
    if len(title) > max_len - 3:
        title = title[:max_len - 6] + "..."
    text_len = max([len(line) for line in lines])
    title_len = len(title)
    line_len = min(max_len, max(title_len, text_len))

    # if each line is longer than max_len, break it into multiple lines
    new_lines = []
    for line in lines:
        while len(line) > line_len:
            new_lines.append(line[:line_len])
            line = line[line_len:]
        new_lines.append(line)
    lines = new_lines

    bar_len = line_len - len(title)
    front_bar_len = bar_len // 2
    back_bar_len = bar_len - front_bar_len
    print(box_color+"╭─" + "─"*front_bar_len + title + "─"*back_bar_len + "─╮"+color.end)
    for line in lines:
        print(box_color+"│ " + text_color + line.ljust(line_len) + box_color + " │"+color.end)
    print(box_color+"╰" + "─" * (line_len + 2) + "╯"+color.end)

def print_warning(*args) -> None:
    text = ' '.join(map(str, args))
    print(color.yellow + color.bold + '[Warning] ' + color.end + color.yellow + text + color.end)

def print_info(*args) -> None:
    text = ' '.join(map(str, args))
    print(color.green + color.bold + '[Info] ' + color.end + color.green + text + color.end)

def print_error(*args) -> None:
    text = ' '.join(map(str, args))
    print(color.red + color.bold + '[Error] ' + color.end + color.red + text + color.end)

def print_note(*args) -> None:
    text = ' '.join(map(str, args))
    print(color.cyan + color.bold + '[NOTE] ' + color.end + color.cyan + text + color.end)

def print_wrap(text, max_width=100):
    wrapped_text = textwrap.fill(text, width=max_width)
    print(wrapped_text)


def print_qna(question, response):
    print("************************************")
    print_wrap(f"*** [QUESTION]\n{question}")
    print("************************************")
    print_wrap(f"*** [RESPONSE]\n{response}")
    print("************************************")
