from IPython.core.magic import register_cell_magic


@register_cell_magic
def mypy(line, cell):

    from IPython import get_ipython
    from mypy import api

    # inserting a newline at the beginning of the cell
    # ensures mypy's output matches the the line
    # numbers in jupyter

    cell = '\n' + cell

    mypy_result = api.run(['-c', cell] + line.split())

    if mypy_result[0]:  # print mypy stdout
        print(mypy_result[0])

    if mypy_result[1]:  # print mypy stderr
        print(mypy_result[1])

    shell = get_ipython()
    shell.run_cell(cell)