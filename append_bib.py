import sys
with open('/Users/albertstarfield/Documents/misc/AdaptiveSystem/project-zephyrine/citations.bib', 'a', encoding='utf-8') as f:
    f.write('\n' + sys.stdin.read() + '\n')
