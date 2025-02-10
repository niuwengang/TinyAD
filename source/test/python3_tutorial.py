class Person:
    def __call__(self, name):
        print("__calll__"+" "+name)
        

    def hello(self,name):
        print(name)


person=Person()
person("zhangsan")
person.hello("zhangsan")