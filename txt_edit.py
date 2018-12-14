import re
with open('parse_text.txt') as file:
    output = open('output.txt', 'w+')
    output_string = ''

    for line in file.readlines():
        var_name = ''
        write_line = '['
        for indx, word in enumerate(line.split()):
            if re.search('[a-zA-Z]', word):
                var_name = word + ' = '
            elif indx == len(line.split()) - 1 and not word == '0':
                write_line += word
            elif not word == '0':
                write_line += word + ', '

        write_line = var_name + write_line + ']' + '\n'
        output_string += write_line
    output.write(output_string)
    output.close()
