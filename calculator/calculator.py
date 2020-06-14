from tkinter import *
import tkinter.font
from functools import partial
def get_input(entry,argu):#除了最后一行按钮，其余按钮的事件处理函数
    entry.insert(END,argu)
def backspace(entry):#按钮"<-"事件处理函数
    input_len=len(entry.get())#get()获取输入内容
    entry.delete(input_len-1)
def clear(entry):#按钮"clear"事件处理函数
    entry.delete(0,END)
def calculate(entry):#按钮"="事件处理函数
    input = entry.get()
    output = str(eval(input.strip()))#eval(input.strip()):eval(str)将字符串str当成有效表达式来求值并返回计算结果(返回int类型)
    clear(entry)
    entry.insert(END, output)
def GUI():
    root=Tk()
    root.title("四则运算计算器")
    root.resizable(0,0)#第一个宽度为False不可改变大小，第二个高度为False不可改变大小
    entry_font=tkinter.font.Font(size=12)#设置输入文本框的字体大小，相应的会改变输入文本框的大小
    entry=Entry(root,justify="right",font=entry_font)#输入框,justify="right"表示多行文本的对齐方式是右对齐
    #网格布局
    entry.grid(row=0,column=0,columnspan=4,sticky=N,padx=5,pady=5)#输入框entry放置在0行0列,columnspan列跨度为4列；
                                                                  #sticky属性表示紧贴单元格的北角,padx/pady表示输入框
                                                                  #entry与窗口root在x和y方向的填充空间大小
    button_font=tkinter.font.Font(size=10,weight=tkinter.font.BOLD)#size为字体大小
                                                                   #weight="bold"(粗体)或"normal"(正常)
                                                                   #slant="italic"(斜体)或"normal"（正常）
                                                                   #underline=1(下划线)或0(正常)
                                                                   #overstrike=1(删除线)或0(正常)
    button_bg="#9932CC"#RGB颜色表，按钮显示为紫色
    button_press_bg='#B3EE3A'#按钮按下时显示的颜色
    myButton=partial(Button,root,bg=button_bg,padx=10,pady=3,activebackground=button_press_bg)
                                                              #partial函数的作用就是：将所作用的函数作为partial（）函数的第一个参数，原函数
                                                              #的各个参数依次作为partial（）函数的后续参数，原函数有关键字参数的一定要带上关
                                                              #键字，没有的话，按原有参数顺序进行补充。（类似于，装饰器decorator，对于函数进
                                                              #行二次包装，产生特殊效果；但又不同于装饰器，偏函数产生了一个新函数，而装饰器，
                                                              #可改变被装饰函数的函数入口地址也可以不影响原函数）。使用偏函数作用在于简化原函数调用

    button7=myButton(text='7',command=lambda : get_input(entry,'7'))
    button7.grid(row=1,column=0,pady=5)#1行0列,pady为按键7与其下面的4按键间隔
    button8=myButton(text='8',command=lambda : get_input(entry,'8'))
    button8.grid(row=1,column=1,pady=5)#1行1列
    button9=myButton(text='9',command=lambda : get_input(entry,'9'))
    button9.grid(row=1,column=2,pady=5)#1行2列
    button_add=myButton(text='+',command=lambda : get_input(entry,'+'))
    button_add.grid(row=1,column=3,pady=5,padx=2)#1行3列
    button4=myButton(text='4',command=lambda : get_input(entry,'4'))
    button4.grid(row=2,column=0,pady=5)#2行0列
    button5=myButton(text='5',command=lambda : get_input(entry,'5'))
    button5.grid(row=2,column=1,pady=5)#2行1列
    button6=myButton(text='6',command=lambda : get_input(entry,'6'))
    button6.grid(row=2,column=2,pady=5)#2行2列
    button_subtract=myButton(text='-',command=lambda : get_input(entry,'-'))
    button_subtract.grid(row=2,column=3,pady=5)#2行3列
    button1=myButton(text='1',command=lambda : get_input(entry,'1'))
    button1.grid(row=3,column=0,pady=5)#3行0列
    button2=myButton(text='2',command=lambda : get_input(entry,'2'))
    button2.grid(row=3,column=1,pady=5)#3行1列
    button3=myButton(text='3',command=lambda : get_input(entry,'3'))
    button3.grid(row=3,column=2,pady=5)#3行2列
    button_multiply=myButton(text='*',command=lambda : get_input(entry,'*'))
    button_multiply.grid(row=3,column=3,pady=5)#3行3列
    button0=myButton(text='0',command=lambda : get_input(entry,'0'))
    button0.grid(row=4,column=0,columnspan=2,pady=5,sticky=N+S+E+W,padx=2)#4行0-1列,x方向与其它空间的距离为3,columnspan常与sticky配合使用,将组件紧贴单元格的某一角
    button_point=myButton(text='.',command=lambda : get_input(entry,'.'))
    button_point.grid(row=4,column=2,pady=5)#4行2列,由于在同一行中，第0列和第1列合并,此处column=2,否则会覆盖
    button_divide=myButton(text='/',command=lambda : get_input(entry,'/'))
    button_divide.grid(row=4,column=3,pady=5)#4行3列
    button_back=myButton(text='<-',command=lambda : backspace(entry))
    button_back.grid(row=5,column=0,pady=5)#5行0列
    button_clear=myButton(text='clear',command=lambda : clear(entry))
    button_clear.grid(row=5,column=1,pady=5)#5行1列
    button_equal=myButton(text='=',command=lambda : calculate(entry))
    button_equal.grid(row=5,column=2,columnspan=2,padx=3,pady=5,sticky=N+S+E+W)#5行2-3列
    root.mainloop()
if __name__ == "__main__":
    GUI()