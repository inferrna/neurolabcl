import traceback

class Watcher(object):
    def __init__(self, obj=None, attr=None, log_file='log.txt', include=[], enabled=False):
        """
            Debugger that watches for changes in object attributes
            obj - object to be watched
            attr - string, name of attribute
            log_file - string, where to write output
            include - list of strings, debug files only in these directories.
               Set it to path of your project otherwise it will take long time
               to run on big libraries import and usage.
        """

        self.log_file=log_file
        with open(self.log_file, 'wb'): pass
        self.prev_st = None
        self.include = [incl.replace('\\','/') for incl in include]
        if obj:
            self.value = getattr(obj, attr)
        self.obj = obj
        self.attr = attr
        self.enabled = enabled # Important, must be last line on __init__.

    def __call__(self, *args, **kwargs):
        kwargs['enabled'] = True
        self.__init__(*args, **kwargs)

    def check_condition(self):
        tmp = getattr(self.obj, self.attr)
        result = tmp != self.value
        self.value = tmp
        return result

    def trace_command(self, frame, event, arg):
        if event!='line' or not self.enabled:
            return self.trace_command
        if self.check_condition():
            print("Condition checked")
            if self.prev_st:
                print("Change detected")
                with open(self.log_file, 'ab') as f:
                    print("Value of",self.obj,".",self.attr,"changed!", file=f)
                    print("###### Line:", file=f)
                    print(''.join(self.prev_st), file=f)
        if self.include:
            fname = frame.f_code.co_filename.replace('\\','/')
            to_include = False
            for incl in self.include:
                if fname.startswith(incl):
                    to_include = True
                    break
            if not to_include:
                return self.trace_command
        self.prev_st = traceback.format_stack(frame)
        return self.trace_command
import sys
watcher = Watcher()
sys.settrace(watcher.trace_command)
