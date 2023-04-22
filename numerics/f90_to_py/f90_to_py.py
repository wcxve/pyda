"""
    Fortran 90 to Python conversion by string manipulations

    Nikolai Shokhirev http://www.numericalexpert.com/blog/fotran90_to_python/

    Simple Fortran 90 to Python conversion by string manipulations:

        Replacement of Fortran statements with corresponding Python statements
        Replacement of math functions
        Commenting out Fortan-specific statment (e.g. integer, end do)
        Conversion of do loops to for loops: do j=1, M, 2 => for j in range(0, M, 2)
        Covertion of array access from ( ) to [ ]
        Conversion of function headers (possibly multi-line)

    The resulting file still require some manual cleanup and formatting.
"""

### Library ###

# List of substitution pairs for statements.
map_list=[('!','#'),
    ('\t','  '),
    ('(/','['),
    ('/)',']'),
    ('.d0','.0'),
    ('d0','0'),
    ('.eq.',' == '),
    ('.ne.',' != '),
    ('/=',' != '),
    ('.lt.',' < '),
    ('.gt.',' > '),
    ('.le.',' <= '),
    ('.ge.',' >= '),
    ('then',':'),
    ('else if','elif'),
    ('else','else:'),
    ('&','\\'),
    ('.and.',' and '),
    ('.or.',' or '),
    ('.true.',' True '),
    ('.false.',' False '),
    ('end if','#end if'),
    ('do while','while'),
    ('end do',''),
    ('end program','# end program'),
    ('end function','# end function'),
    ('end subroutine','# end subroutine'),
    ('end','# end'),
    ('program','def'),
    ('function','def'),
    ('subroutine','def'),
    ('integer','#integer'),
    ('real','#real'),
    ('allocate','#allocate'),
    ('pause','#pause'),
    ]

# List of substitution pairs for math functions (incomplete).
math_list=[
    ('dabs','abs'),
    ('dsqrt','math.sqrt'),
    ('dlog','math.log'),
    ('dexp','math.exp'),
    ]

def file2list(fname, include_eol=False):
    ''' text file to list of string '''
    with open(fname, encoding="utf8") as f:
        if include_eol:
            content = f.readlines() # \n is included
        else:
            content = f.read().splitlines() # \n is not included
        return content

def ireplace(old, new, text):
    ''' Case Insensitive Replace excluding comments
    based on
    http://stackoverflow.com/questions/919056/python-case-insensitive-replace
    ''' 
    idx = 0
    lim = text.find('#')
    if lim < 0:
        lim = len(text)
    while idx < lim:
#     while idx < len(text):
        index_l = text.lower().find(old.lower(), idx)
        if index_l == -1:
            return text
        text = text[:index_l] + new + text[index_l + len(old):]
        idx = index_l + len(old)
    return text

def replace_statements(content,map_list):
    ''' replaice all statements in map_list for all lines '''
    result = list(content) # make a copy
    for p in map_list:
#         result = [l.replace(p[0], p[1]) for l in result] # Case-sensitive
        result = [ireplace(p[0], p[1], l) for l in result]
    return result

def replace_math_functions(content,math_list):
    ''' replaice all math functions in content '''
    for p in math_list:
        content = [ireplace(p[0], p[1], l) for l in content]
    return content

def process_do(line):
    ''' replaice Fortran do to Python for:
        do i=1, N => for i in range(0, N):
        do j=1, M, 2 => for j in range(0, M, 2):
        do iq=ip+1, N => for iq in range(ip+1, N):
    '''
    if line.startswith('#'):
        return line
    if line.strip().startswith( 'do' ):
        if '#' in line:
            do_var_lims,comment = line.split('#')
            comment = ' #'+''.join(comment)
        else:
            do_var_lims,comment = line, ''
        do_var,lims = do_var_lims.split('=')
        for_var = do_var.replace('do','for') + ' in range('
        lims=lims.split(',')
        lims=[i.strip() for i in lims]
        if str(lims[0])=='1':     # heuristic correction for 0-based arrays    
            lims[0]='0'           # warning: this can be unnecessery sometimes.
        result = for_var + ', '.join(lims) + '):'
        if comment:
            result += comment
        return result
    else:
        return line

def replace_all_do(content):
    ''' replaice do statements for all lines '''
    result = [process_do(l) for l in content]
    return result

def adjust_array(line, arr):
    ''' Conversion of Fortran array access to Python:
        line = 'A(i,j) = A(m,A(k,l))'
        arr = 'A'
        result = 'A[i,j] = A[m,A[k,l]]'
    '''
    if line.startswith('#'):
        return line
    i = line.find(arr+'(')
    while i >= 0:
        j = line.find('(',i)
        line = line[:j] + '[' + line[j+1:]
        c = 1
        for k in range(j+1,len(line)):
            if line[k] == '(':
                c += 1
            if line[k] == ')':
                c -= 1
            if c == 0:
                line = line[:k] + ']' + line[k+1:]
#                 i = line.find(arr+'(', k+1) # does not process nested arrays
                i = line.find(arr+'(')
                break
    return line

def adjust_arrays(line, arrays):
    ''' adjust all arrays in a line '''
    for a in arrays:
        line = adjust_array(line, a)
    return line

def adjust_all_arrays(content, arrays):
    ''' adjust all arrays in all lines '''
    result = [adjust_arrays(l, arrays) for l in content]
    return result

def adjust_functions(content):
    """ Adds ':' after ')' """    
    for n,line in enumerate(content):
        count = 0
        if line.strip().startswith('def'):
            i = line.find('(')
            if i >= 0:
                count = 1
                for k in range(i,len(line)):
                    #print(k, line[k])
                    if line[k] == '#':
                        break
                    if line[k] == ')':
                        count = 0
                        content[n] = line[:k+1] + ':' + line[k+1:]
                        break
            else:
                count = 0
                i = line.find('#') 
                if i >= 0:
                    content[n] =  line[:i] + ':' + line[i:]
                else:
                    content[n] =  line + ':'
        else:
            if count > 0:
                i = 0
                for k in range(i,len(line)):
                    if line[k] == '#':
                        break
                    if line[k] == ')':
                        count = 0
                        content[n] = line[:k+1] + ':' + line[k+1:]
                        break
    return content

def list2file(fname, output):
    ''' list of string to text file '''
    with open(fname,'w', encoding="utf8") as out:
        for line in output:
            out.write(line+'\n')

def f90_to_py(content,map_list,math_list,arrays):
    ''' Fortran 90 to Python conversion 
        by string manipulations
        content - input list of f90 file content strings
        map_list - list of substitution pairs, 
                   e.g. ('.lt.',' < '),('DABS','abs'),
        output - list of python file content strings
    '''
    output = replace_statements(content,map_list)
    output = replace_math_functions(output,math_list)
    output = replace_all_do(output)
    if arrays:
        output = adjust_all_arrays(output, arrays)
    output = adjust_functions(output)
    return output
    

	

if __name__ == "__main__":	
    import sys
    if len(sys.argv) < 3:
        print("Usage: \npython f90_to_py.py input_file_name output_filename \"array_list\"")
        print("Example: \npython f90_to_py.py ujacobi.f90 ujacobi.py \"A,D,V,B,Z\"")
    else:
        fname = sys.argv[1]
        foutname = sys.argv[2]
        if len(sys.argv) > 3:
            arrays = sys.argv[3].split(",")
        else:
            arrays = []
        content = file2list(fname)
        output = f90_to_py(content,map_list,math_list,arrays)
        list2file(foutname, output)
