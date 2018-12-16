with open('parse_text.txt') as input:
    output = open('output.txt', 'w+')
    output_string = ''

    for line in input.readlines():
        var_name = ''
        write_line = '['
        for indx, word in enumerate(line.split()):
            if word == 'and':
                continue
            # remove , . and {}
            filt_word = word
            filt_word = filt_word.replace(',', '')
            filt_word = filt_word.replace('{', '[')
            filt_word = filt_word.replace('}', ']')
            filt_word = filt_word.replace('.', '')

            word_list = ['A', 'B', 'C', 'D']
            if filt_word in word_list:
                var_name = filt_word
            elif indx == len(line.split()) - 1:
                write_line += filt_word
            else:
                write_line += filt_word + ', '
        write_line = var_name + ' = ' + write_line + ']' + '\n'
        output_string += write_line
    output.write(output_string)            
    output.close()

