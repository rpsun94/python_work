"""Some functions."""


import numba
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from RSDttest import RSDttest
from const import Pi
import scipy.stats


class Reader():
    def __init__(self, file_path, file_namels, splittext=None, number=1):
        self.path = file_path
        self.namels = file_namels
        self.splittext = splittext
        self.num = number

    def spliter(self):
        data = {}
        for name in self.namels:
            data[name] = []
        code_type_list = ['ansi', 'utf-8', 'gbk']
        if self.isexist():
            if self.isansi():
                ct = 0
            elif self.isutf():
                ct = 1
            elif self.isgbk():
                ct = 2
            with open(self.path, 'r', encoding=code_type_list[ct]
                      )as file_object:
                n = 0
                if self.splittext is None:
                    for line in file_object:
                        if n > self.num - 2:
                            ls_line = line.split()
                            if len(ls_line) != len(self.namels):
                                msg = 'Variable Number Not Match!'
                                print(msg)
                                break
                            for i in range(0, len(self.namels)):
                                try:
                                    data[self.namels[i]].append(
                                        float(ls_line[i]))
                                    # 一行代码最好不要超过72个字符,在合适的位置换
                                    # 行是个不错的主意,但不要忘记缩进
                                except ValueError:
                                    data[self.namels[i]].append(
                                        ls_line[i])
                        n = n + 1
                if not (self.splittext is None):
                    for line in file_object:
                        if n > self.num - 2:
                            ls_line = line.split(self.splittext)
                            if len(ls_line) != len(self.namels):
                                msg = 'Variable Number Not Match!'
                                print(msg)
                                break
                            for i in range(0, len(self.namels)):
                                try:
                                    data[self.namels[i]].append(
                                        float(ls_line[i]))
                                except ValueError:
                                    data[self.namels[i]].append(
                                        ls_line[i])
                        n = n + 1
        return data

    def isexist(self):
        l_exist = True
        try:
            with open(self.path, 'r') as file_object:
                l_try = True
        except FileNotFoundError:
            msg = 'Sorry,' + self.path + ' cant be found.'
            print(msg)
            l_exist = False
        return l_exist

    def isutf(self):
        l_utf = True
        try:
            with open(self.path, 'r', encoding='utf-8')as file_object:
                temp = file_object.read()
        except UnicodeDecodeError:
            print('Not utf-8')
            l_utf = False
        return l_utf

    def isansi(self):
        l_ansi = True
        try:
            with open(self.path, 'r', encoding='ansi')as file_object:
                temp = file_object.read()
        except UnicodeDecodeError:
            print('Not ANSI')
            l_ansi = False
        return l_ansi

    def isgbk(self):
        l_gbk = True
        try:
            with open(self.path, 'r', encoding='gbk')as file_object:
                temp = file_object.read()
        except UnicodeDecodeError:
            print('Not GBK')
            l_gbk = False
        return l_gbk

    def help(self):
        print('you should use this as "XX=Reader(file_path,file_namels,'
              + 'splittext,number)"\nyou should plus "r" before your path '
              + 'like r' + "'" + "E:" + '\\' + '1660.dat' + "'"
              + ' to avoid error\nfile_namels'
              + ' means names of variable\nif splittext==None it will use '
              + '.split(),or it will use .split(splittext)\nnumber means '
              + 'the line starts'
              )


def datelist(start, end, step=None, remodel=None):
    """Make datelist.

    Parameters
    ----------
    start : tuple or list like (2016,2,3)
        Date started
    end : tuple or list like (2016,2,10)
        Date ended
    step : str, optional
        Step.'year', 'month', 'day' or 'hours=6'. The default is day.
    remodel : str, optional
        'str': return str.
        'date': return date type.
        The default is str.

    Returns
    -------
    result : datelist

    """
    # input like xx=datelist((2016,2,3),(2018,3,20))
    # return date list from start to end as str
    # like ['20160203','20160204',...]
    import datetime
    import re
    start_date = datetime.datetime(*start)
    end_date = datetime.datetime(*end)

    result = []
    curr_date = start_date
    if step is None or step.lower() == 'day':
        fmt = "%04d%02d%02d"
        fmt1 = "%Y%m%d"
    elif step.lower() == 'month':
        fmt = "%04d%02d"
        fmt1 = "%Y%m"
    elif step.lower() == 'year':
        fmt = "%04d"
        fmt1 = "%Y"
    if step is None or step.lower() == 'day':
        while curr_date != end_date:
            result.append(fmt % (curr_date.year, curr_date.month,
                                 curr_date.day))
            curr_date += datetime.timedelta(1)
        result.append(fmt % (curr_date.year, curr_date.month,
                             curr_date.day))

    elif step.lower() == 'month':
        result.append(fmt % (curr_date.year, curr_date.month))
        old_month = curr_date.month
        while curr_date != end_date:
            if curr_date.month != old_month:
                result.append(fmt % (curr_date.year, curr_date.month))
                old_month = curr_date.month
            curr_date += datetime.timedelta(1)
        if curr_date.month != old_month:
            result.append(fmt % (curr_date.year, curr_date.month))

    elif step.lower() == 'year':
        result.append(fmt % (curr_date.year))
        old_year = curr_date.year
        while curr_date != end_date:
            if curr_date.year != old_year:
                result.append(fmt % (curr_date.year))
                old_year = curr_date.year
            curr_date += datetime.timedelta(1)
        if curr_date.year != old_year:
            result.append(fmt % (curr_date.year))
    else:
        step_ele = re.search('(\S+)=\S+', step).group(1)
        step_num = re.search('\S+=(\S+)', step).group(1)
        dic = {
            step_ele: int(step_num),
        }
        while curr_date != (end_date + datetime.timedelta(**dic)):
            result.append(curr_date)
            curr_date += datetime.timedelta(**dic)
            # print(curr_date)
        # result.append(curr_date)
        return result
        exit()
    if remodel is None or remodel.lower() == 'str':
        pass
    elif remodel.lower() == 'date':
        dates = []
        for time in result:
            current_date = datetime.datetime.strptime(time, fmt1)
            dates.append(current_date)
        result = dates
    return result


def mean_data(data, option):
    import numpy
    data_test = data.reshape(-1)[0]
    dtype = str(type(data_test))[14:-2]
    while not dtype.isalpha():
        dtype = dtype[0:-1]
    shape_data = data.shape
    order = option['order']
    standard = option['standard']
    time_range = option['range']
    index = len(order) + 1
    for i in range(0, len(order)):
        if order[i].lower() == standard.lower():
            index = i
    list0 = list(range(0, len(order)))
    list1 = []
    list1.append(index)
    for i in list0:
        if i != index:
            list1.append(i)
    data_trans = data.transpose(list1)
    shape_trans = data_trans.shape
    list2 = shape_trans[2::]
    answer = numpy.ones(list2, dtype=dtype)
    temp = numpy.ones(list2, dtype=dtype)
    answer = answer - temp
    if len(time_range) <= shape_data[index]:
        for i in time_range:
            answer = answer + data_trans[i, :]
    answer = answer / len(time_range)
    return answer


@numba.jit(nopython=True)
def interp2(img, shape0, mean=True):
    """Interp2 as matlab.

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    shape : tuple
        DESCRIPTION.

    Returns
    -------
    emptyImage : TYPE
        DESCRIPTION.

    """
    m = shape0[0]
    n = shape0[1]
    height, width = img.shape
    # logic_do = False
    emptyImage = numpy.ones((m, n), dtype=numba.float64) - 1
    if m >= height and n >= width:
        # logic_do = True
        sh = (m - 1.0) / (height - 1.0)
        sw = (n - 1.0) / (width - 1.0)
        img_alter = numpy.ones((height + 1, width + 1), dtype=numba.float64)
        img_alter[0:height, 0:width] = img[:, :].copy()
        img_alter[height, 0:width] = img[-1, :]
        img_alter[0:height, width] = img[:, -1]
        img_alter[height, width] = img[-1, -1]
        for i in numba.prange(m):
            for j in numba.prange(n):
                x = i / sh
                y = j / sw
                p = x - numpy.int(x)
                q = y - numpy.int(y)
                x = numpy.int(x)
                y = numpy.int(y)
                ii = img_alter[x:x+2, y:y+2]
                ip = numpy.array([[1-p, 1-p], [p, p]])
                iq = numpy.array([[1-q, q], [1-q, q]])
                value = numpy.sum(ii*ip*iq)
                emptyImage[i, j] = value
    if m < height and n < width and mean:
        # logic_do = True
        sh = (height - 1.0) / (m - 1.0)
        sw = (width - 1.0) / (n - 1.0)
        h_alter = numpy.arange(0, height - 1 + 0.1, sh)
        w_alter = numpy.arange(0, width - 1 + 0.1, sw)
        # print(h_alter)
        # print(w_alter)
        # h=np.arange(0,height+0.1)
        # w=np.arange(0,width+0.1)
        for i in numba.prange(m):
            for j in numba.prange(n):
                h_up = sh / 2
                h_down = sh / 2
                w_left = sw / 2
                w_right = sw / 2
                if i == 0:
                    h_up = 0
                if i == m - 1:
                    h_down = 0
                if j == 0:
                    w_left = 0
                if j == n - 1:
                    w_right = 0
                hst = numpy.ceil(h_alter[i] - h_up)
                hed = numpy.int(h_alter[i] + h_down)
                wst = numpy.ceil(w_alter[j] - w_left)
                wed = numpy.int(w_alter[j] + w_right)
                # print(hst,hed,wst,wed)
                emptyImage[i, j] = numpy.mean(img[hst:hed + 1, wst:wed + 1])
    if m < height and n < width and (not mean):
        # logic_do = True
        sh = (height - 1.0) / (m - 1.0)
        sw = (width - 1.0) / (n - 1.0)
        h_alter = numpy.arange(0, height - 1 + 0.1, sh)
        w_alter = numpy.arange(0, width - 1 + 0.1, sw)
        # print(h_alter)
        # print(w_alter)
        # h=np.arange(0,height+0.1)
        # w=np.arange(0,width+0.1)
        for i in numba.prange(m):
            for j in numba.prange(n):
                h_up = sh / 2
                h_down = sh / 2
                w_left = sw / 2
                w_right = sw / 2
                if i == 0:
                    h_up = 0
                if i == m - 1:
                    h_down = 0
                if j == 0:
                    w_left = 0
                if j == n - 1:
                    w_right = 0
                hst = numpy.ceil(h_alter[i] - h_up)
                hed = numpy.int(h_alter[i] + h_down)
                wst = numpy.ceil(w_alter[j] - w_left)
                wed = numpy.int(w_alter[j] + w_right)
                # print(hst,hed,wst,wed)
                emptyImage[i, j] = numpy.sum(img[hst:hed + 1, wst:wed + 1])
    # if not logic_do:
    #     print('you should do interp2 for 2 time')
    #     exit()
    return emptyImage


def mkdir(path):
    """Mkdir.

    Parameters
    ----------
    path : TYPE
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' is created successfully')
        return True
    elif isExists:
        print(path + ' is exist')
        return False


def getcolorbar(name=None, reverse=False):
    """Getcolorbar.

    Parameters
    ----------
    name : TYPE, optional
        DESCRIPTION. The default is None.
    reverse : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    from demo_function import Reader
    import matplotlib.colors as col
    import json
    if name is None:
        name = 'BlGrYeOrReVi200'
    path = r'D:\python_work\colormap'
    control_file = open(path + '\\colorbar.json')
    control_dict = json.load(control_file)
    number = control_dict[name]['start_line']
    control = control_dict[name]['control_num']
    fl = Reader(path + '\\' + name + '.rgb', ['r', 'g', 'b'],
                number=number)
    # http://www.ncl.ucar.edu/Document/Graphics/ColorTables/BlGrYeOrReVi200.shtml
    rgb = fl.spliter()
    # print(rgb)
    num = len(rgb['r'])
    # print(num)
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
    }
    if reverse:
        def position(i):
            return 0

        def func(i):
            y = (num - 1 - i) / (num - 1)
            return y
    elif not reverse:
        def position(i):
            return i

        def func(i):
            y = i / (num - 1)
            return y
    for i in range(0, num):
        cdict['red'].insert(position(i),
                            (func(i), float(rgb['r'][i] / control),
                             float(rgb['r'][i] / control)))
        cdict['green'].insert(position(i),
                              (func(i), float(rgb['g'][i] / control),
                               float(rgb['g'][i] / control)))
        cdict['blue'].insert(position(i),
                             (func(i), float(rgb['b'][i] / control),
                              float(rgb['b'][i] / control)))
    # print(cdict['red'])
    my_cmap = col.LinearSegmentedColormap(name='newcmap',
                                          segmentdata=cdict, N=256)
    return my_cmap


def auto_split_season(data1, date, seasons, seasonname=None, align=False, re='mean'):
    """Func to split season.

    Parameters
    ----------
    data1 : TYPE
        DESCRIPTION.
    date : TYPE
        DESCRIPTION.
    seasons : TYPE
        DESCRIPTION.
    seasonname : TYPE, optional
        DESCRIPTION. The default is None.
    align : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    dic_alter : TYPE
        DESCRIPTION.

    """
    # data is an 1D-array or array_like
    # season is a list telling which months make a season
    # seasonname can be ignore
    # return a dic contain season sum
    # this method may lose some data
    # date should be an datelist
    # in season ,0 means Jan,and should be in order
    import numpy
    if seasonname is None:
        seasonname = [str(i) for i in range(len(seasons))]
    dic = {}
    data = numpy.array(data1)
    if re == 'mean':
        for i in range(len(seasons)):
            season = seasons[i]
            nmonth = len(season)
            temp_date = []
            temp_data = []
            mean_data = []
            count = -1
            start_count = 0
            for j in range(len(date)):
                if (date[j].month - 1) in season:
                    temp_date.append(date[j])
                    temp_data.append(data[j])
                    count += 1
                    if (count < nmonth) and (date[j].month == (season[0] + 1)):
                        start_count = count
            j = start_count
            mean_date = []
            mean_data = []
            while (j + nmonth) <= len(temp_date):
                mean_date.append(temp_date[j])
                mean_data.append(numpy.mean(temp_data[j:j + nmonth],axis=0))
                j += nmonth
            dic[seasonname[i]] = {
                'date': mean_date,
                'data': mean_data,
            }
    elif re == 'sum':
        for i in range(len(seasons)):
            season = seasons[i]
            nmonth = len(season)
            temp_date = []
            temp_data = []
            mean_data = []
            count = -1
            start_count = 0
            for j in range(len(date)):
                if (date[j].month - 1) in season:
                    temp_date.append(date[j])
                    temp_data.append(data[j])
                    count += 1
                    if (count < nmonth) and (date[j].month == (season[0] + 1)):
                        start_count = count
            j = start_count
            mean_date = []
            mean_data = []
            while (j + nmonth) <= len(temp_date):
                mean_date.append(temp_date[j])
                mean_data.append(numpy.sum(temp_data[j:j + nmonth], axis=0))
                j += nmonth
            dic[seasonname[i]] = {
                'date': mean_date,
                'data': mean_data,
            }
    start_year = int(max([dic[name]['date'][0].year for name in seasonname]))
    end_year = int(min([dic[name]['date'][-1].year for name in seasonname]))
    dic_alter = {}
    if not align:
        dic_alter = dic
    elif align:
        for key in list(dic.keys()):
            for i in range(len(dic[key]['date'])):
                if dic[key]['date'][i].year == start_year:
                    start = i
                if dic[key]['date'][i].year == end_year:
                    end = i
            dic_alter[key] = {
                'date': dic[key]['date'][start:end + 1],
                'data': dic[key]['data'][start:end + 1],
            }

    return dic_alter


def get_para(X0):
    """get_param.

    Parameters
    ----------
    X0 : TYPE
        DESCRIPTION.

    Returns
    -------
    X0 : TYPE
        DESCRIPTION.

    """
    for j in range(X0.shape[0]):
        X0[j, :] -= numpy.mean(X0[j, :])
        std = numpy.std(X0[j, :])
        if std != 0:
            X0[j, :] /= std
    return X0


def add_one(keys, value, limit_min=-9999, limit_max=9999):
    """add_one.

    Parameters
    ----------
    keys : TYPE
        DESCRIPTION.
    value : TYPE
        DESCRIPTION.
    limit_min : TYPE, optional
        DESCRIPTION. The default is -9999.
    limit_max : TYPE, optional
        DESCRIPTION. The default is 9999.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """
    nkeys = len(keys)+1
    data = numpy.array([0]*(nkeys), dtype='float64')
    loop = numpy.arange(0, nkeys, 1, dtype='int64')
    for i in loop:
        if i == 0:
            v_l, v_r = limit_min, keys[i]
        elif 0 < i < nkeys-1:
            v_l, v_r = keys[i-1], keys[i]
        elif i == nkeys-1:
            v_l, v_r = keys[i-1], limit_max
        if v_l <= value < v_r:
            data[i] += 1
            break
    return data


def paint_figure(lat, lon, basemap_option, name, save=True,
                 contourf=False, data_contourf=None,
                 scatter=False, data_scatter=None,
                 scatter_latlon=False, data_scatter_latlon=None,
                 multi_scatter=False, data_multi_scatter=None,
                 contour=False, data_contour=None,
                 quiver=False, data_quiver=None,
                 pcolor=False, data_pcolor=None,
                 plot=False, data_plot=None,
                 title=False, nobar=False, vector='vertical',titlesize=30,
                 dpi=300,rotation=False,fz = 40):
    """


    Parameters
    ----------
    lat : TYPE
        DESCRIPTION.
    lon : TYPE
        DESCRIPTION.
    basemap_option : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.
    save : TYPE, optional
        DESCRIPTION. The default is True.
    contourf : TYPE, optional
        DESCRIPTION. The default is False.
    data_contourf : TYPE, optional
        DESCRIPTION. The default is None.
    scatter : TYPE, optional
        DESCRIPTION. The default is False.
    data_scatter : TYPE, optional
        DESCRIPTION. The default is None.
    contour : TYPE, optional
        DESCRIPTION. The default is False.
    data_contour : TYPE, optional
        DESCRIPTION. The default is None.
    quiver : TYPE, optional
        DESCRIPTION. The default is False.
    data_quiver : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    # data_contourf={'0': {'data':... ,'cmap': ...,'levels':...,},
    #               '1': {},...
    #               }
    # data_contour={'0': {'data':... ,'colors': ...,'levels':...,},
    #               '1': {},...
    #               }
    # data_scatter={'0': {'data': ...,'colors':... ,'logic':('data<0.05')...,},
    #               '1': {},...
    #               }
    # data_scatter_latlon={'0': {'data':(lat,lon),'colors':...,},
    #               '1': {},...
    #               }
    # data_quiver={'0': {'data':(complex)... ,'standard_wind': ...,},
    #               '1': {},...
    #               }
    # define param============================================================
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    if vector == 'vertical':
        houkou = 'right'
    elif vector == 'horizontal':
        houkou = 'left'
    font = {'family' : 'SimHei',
            'color'  : 'k',
            'weight' : 'normal',
            'size'   : fz,
            }
    M, N = len(lon), len(lat)
    xx = numpy.ones([N, M]) - 1
    yy = numpy.ones([N, M]) - 1
    for index, ndata in numpy.ndenumerate(xx):
        iy, ix = index
        xx[index] = lon[ix]
        yy[index] = lat[iy]
    k = 1.5
    kdpi = dpi/100
    figsize = (12*k, 12)
    fig1 = plt.figure(1, figsize=figsize,dpi=dpi)
    ax = fig1.add_subplot(111)
    m = Basemap(**basemap_option)
    meridians = numpy.arange(0., 360., 90.)
    parallels = numpy.arange(-90., 91., 30.)
    x, y = m(xx, yy)
    m.drawparallels(parallels, labels=[1, 1, 1, 1], fontsize=fz, linewidth=0)
    if not rotation:
        m.drawmeridians(meridians, labels=[1, 1, 1, 1],
                        fontsize=fz, linewidth=0)
    elif rotation:
        items = m.drawmeridians(meridians, labels=[1, 1, 1, 1],
                                fontsize=fz, linewidth=0, textcolor='white',xoffset=300000,yoffset=300000)
        for item in items:
            text = items[item][1][0]
            ax.text(s=text._text,
                    x=text._x,
                    y=text._y,
                    rotation=item,
                    fontsize=fz,
                    ha='center',
                    va='center')
    m.drawcoastlines(color='gray')
    m.drawmapboundary()
    if contourf:
        keys = list(data_contourf.keys())
        CS = []
        for key in keys:
            data = data_contourf[key]['data']
            key_ = list(data_contourf[key].keys())
            if 'mask' in key_:
                mask = data_contourf[key]['mask']
                data = numpy.ma.array(data=data, mask=mask)
            levels = data_contourf[key]['levels']
            cmap = data_contourf[key]['cmap']
            cs = m.contourf(x, y, data, cmap=cmap,
                            levels=levels, extend='both')
            if 'hatches' in key_:
                data_hatches = data_contourf[key]['data_hatches']
                if 'mask' in key_:
                    mask = data_contourf[key]['mask']
                    data_hatches = numpy.ma.array(data=data_hatches, mask=mask)
                hatches = data_contourf[key]['hatches']
                hatches.insert(0, '')
                nz = len(hatches)+1
                a = m.contourf(x, y, data_hatches, nz,
                    hatches = hatches,alpha=0.01)
            divider = make_axes_locatable(ax)
            size_percent = 0.05
            percent = str(size_percent*100) + '%'
            pp = 1
            if basemap_option['projection'] == 'npstere':
                pp = 1.5
            CS.append(cs)

    if contour:
        keys = list(data_contour.keys())
        for key in keys:
            data = data_contour[key]['data']
            key_ = list(data_contour[key].keys())
            if 'mask' in key_:
                mask = data_contour[key]['mask']
                data = numpy.ma.array(data=data, mask=mask)
            levels = data_contour[key]['levels']
            colors = data_contour[key]['colors']
            cs2 = m.contour(x, y, data, levels=levels, colors=colors,
                            linewidths=2)
            if 'clabel' in key_:
                if data_contour[key]['clabel']:
                    fmt='%1.3f'
                    if 'fmt' in key_:
                        fmt = data_contour[key]['fmt']
                    plt.clabel(cs2, inline=True, fontsize=fz*0.75,fmt=fmt)
    if scatter:
        keys = list(data_scatter.keys())
        for key in keys:
            data_ = data_scatter[key]['data']
            key_ = list(data_scatter[key].keys())
            if 'mask' in key_:
                mask = data_scatter[key]['mask']
                data = numpy.ma.array(data=data, mask=mask)
            logic = data_scatter[key]['logic']
            colors = data_scatter[key]['colors']
            dlat = abs(lat[1]-lat[0])
            dlon = abs(lon[1]-lon[0])
            skip_step_y = numpy.max((int(1/dlat+1), 1))
            skip_step_x = numpy.max((int(2/dlon), 1))
            skip = (slice(None, None, skip_step_y),
                    slice(None, None, skip_step_x))
            data = data_[skip]
            x_ = x[skip]
            y_ = y[skip]
            m.scatter(x_[numpy.where(eval(logic))],
                      y_[numpy.where(eval(logic))],
                      s=5*kdpi, c=colors)
    if multi_scatter:
        data = data_multi_scatter['data']
        key_ = list(data_multi_scatter.keys())
        if 'mask' in key_:
            mask = data_multi_scatter['mask']
            data = numpy.ma.array(data=data, mask=mask)
        colors = data_multi_scatter['colors']
        n = len(colors)
        labels = [None]*n
        if 'labels' in key_:
            labels = data_multi_scatter['labels']
        A = []
        for i in range(n):
            a = m.scatter(x[numpy.where(data == i)],
                           y[numpy.where(data == i)],
                           s=5*kdpi, c=colors[i], label=labels[i])
            A.append(a)

    if plot:
        keys = list(data_plot.keys())
        for key in keys:
            #print(key)
            key_ = list(data_plot[key].keys())
            lon = data_plot[key]['lon']
            lat = data_plot[key]['lat']
            option = {}
            option['c'] = data_plot[key]['color']
            option['linewidth'] = data_plot[key]['linewidth']
            if 'latlon' in key_:
                option['latlon'] = data_plot[key]['latlon']
            if 'zorder' in key_:
                option['zorder'] = data_plot[key]['zorder']
            if 'alpha' in key_:
                option['alpha'] = data_plot[key]['alpha']
            m.plot(lon, lat, **option)

    if scatter_latlon:
        keys = list(data_scatter_latlon.keys())
        A = []
        for key in keys:
            # print(key)
            key_ = list(data_scatter_latlon[key].keys())
            data_ = data_scatter_latlon[key]['data']
            colors = data_scatter_latlon[key]['colors']
            label=None
            if 'label' in key_:
                label = data_scatter_latlon[key]['label']
            marker = '.'
            if 'marker' in key_:
                marker = data_scatter_latlon[key]['marker']
            s = 10
            if 's' in key_:
                s = data_scatter_latlon[key]['s']
            latlon=True
            if 'latlon' in key_:
                latlon = data_scatter_latlon[key]['latlon']
            zorder=1
            if 'zorder' in key_:
                zorder = data_scatter_latlon[key]['zorder']
            alpha = 1
            if 'alpha' in key_:
                alpha = data_scatter_latlon[key]['alpha']
            if len(data_.shape)==2:
                a = m.scatter(data_[:, 1],
                          data_[:, 0],
                          s=s*kdpi, c=colors, latlon=latlon,
                          label=label,marker=marker,zorder=zorder,alpha=alpha)
            elif len(data_.shape)==1:
                a = m.scatter(data_[1],
                          data_[0],
                          s=s*kdpi, c=colors, latlon=latlon,
                          label=label,marker=marker,zorder=zorder,alpha=alpha)
            #print('finished')
            A.append(a)
    if quiver:
        keys = list(data_quiver.keys())
        for key in keys:
            data = data_quiver[key]['data']
            key_ = list(data_quiver[key].keys())
            if 'mask' in key_:
                mask = data_quiver[key]['mask']
                data = numpy.ma.array(data=data, mask=mask)
            standard_wind = data_quiver[key]['standard_wind']
            dlat = abs(lat[1]-lat[0])
            dlon = abs(lon[1]-lon[0])
            skip_step_y = numpy.max((int(1/dlat+1), 1))
            skip_step_x = numpy.max((int(2/dlon), 1))
            skip = (slice(None, None, skip_step_y),
                    slice(None, None, skip_step_x))
            u = data.real
            v = data.imag
            u, v, x, y = m.rotate_vector(u, v, xx, yy, returnxy=True)
            cs3 = m.quiver(x[skip], y[skip], u[skip], v[skip], units='dots',
                           width=4, color='k', headlength=5,
                           scale=standard_wind, minshaft=2, minlength=0.1)
            fz = {'size': 20, }
            plt.quiverkey(cs3, 0.75, 1.01, 4*standard_wind,
                          label=':'+str(4*standard_wind)+' m/s',
                          labelpos='E', fontproperties=fz)
    if pcolor:
        from matplotlib.colors import BoundaryNorm
        keys = list(data_pcolor.keys())
        for key in keys:
            data_ = data_pcolor[key]['data']
            key_ = list(data_pcolor[key].keys())
            if 'mask' in key_:
                mask = data_pcolor[key]['mask']
                data_ = numpy.ma.array(data=data_, mask=mask)
            cmap = data_pcolor[key]['cmap']
            levels = data_pcolor[key]['levels']
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
            cs4 = m.pcolormesh(x, y, data_, cmap=cmap, norm=norm)
            divider = make_axes_locatable(ax)
            size_percent = 0.05
            percent = str(size_percent*100) + '%'
            pp = 1
            if basemap_option['projection'] == 'npstere':
                pp = 1.5
    if contourf and (not nobar):
        cax = divider.append_axes(houkou, size=percent, pad=pp)
        bar = plt.colorbar(CS[0], cax=cax, orientation=vector)
        bar.ax.tick_params(labelsize=fz)
        if 'label' in key_:
            cblabel = data_contourf[key]['label']
            bar.set_label(cblabel, fontdict=font)
    if pcolor and (not nobar):
        cax = divider.append_axes(houkou, size=percent, pad=pp)
        bar = plt.colorbar(cs4, cax=cax, orientation=vector)
        bar.ax.tick_params(labelsize=fz)
    if multi_scatter:
        if ('labels' in key_):
            plt.legend(handles=A, loc='center right',
                       bbox_to_anchor=(1, 1), fontsize=fz)
    if scatter_latlon:
        if ('label' in key_):
            plt.legend(handles=A, loc='center left',
                       bbox_to_anchor=(1, 1), fontsize=fz)
    if title:
        plt.title(title, fontsize=titlesize,pad=0.05*figsize[1]*100)
    if save:
        print('saving')
        plt.savefig(name, dpi=fig1.dpi, bbox_inches='tight')
    elif not save:
        plt.show()
    if contourf and nobar:
        if vector == 'vertical':
            size_bar = (figsize[0]*size_percent, figsize[1])
        elif vector == 'horizontal':
            size_bar = (figsize[0], figsize[1]*size_percent,)
        fig2 = plt.figure(2, figsize=size_bar, dpi=dpi)
        cax = fig2.add_subplot(111)
        bar = plt.colorbar(CS[0], cax=cax, orientation=vector)
        if 'label' in key_:
            cblabel = data_contourf[key]['label']
            bar.set_label(cblabel, fontdict=font)
        bar.ax.tick_params(labelsize=fz)
        plt.savefig(name+'_bar.png',dpi=fig2.dpi, bbox_inches='tight')
        plt.close(fig2)
    plt.close(fig1)


def paint_timeseries(x, dic_y, xlabel, ylabel, name,
                     dpi=300, save=True, make_legend=False, str_x=False,
                     loc='center right', figsize=(18, 12), fz=40,
                     minor=False,style='line',rotation='auto',y_lim=False):
    """Paint timeseries


    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    dic_y : dic
    dic_y = {'0':
             {'y':...,
              'color':...,
              'label':...[optional],
              }
             }
        DESCRIPTION.
    xlabel : TYPE
        DESCRIPTION.
    ylabel : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.
    save : TYPE, optional
        DESCRIPTION. The default is True.
    make_legend : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig1 = plt.figure(1, figsize=figsize)
    ax = fig1.add_subplot(111)
    A = []
    keys = numpy.array(list(dic_y.keys()))
    nn = len(keys)
    if not str_x:
        xt = x
    elif str_x:
        xt = numpy.arange(len(x))
    for ikey in range(nn):
        key = keys[ikey]
        dic_y_ = dic_y[key]
        key_ = list(dic_y_.keys())
        y = dic_y_['y']
        color = dic_y_['color']
        label = None
        linestyle = 'o-'
        if 'linestyle' in key_:
            linestyle = dic_y_['linestyle']
        linewidth = '5'
        if 'linewidth' in key_:
            linewidth = dic_y_['linewidth']
        width=0.8
        if 'width' in key_:
            width=dic_y_['width']
        x_ = xt
        if 'x' in key_:
            x_ = dic_y_['x']
        if 'label' in key_:
            label = dic_y_['label']
        if style=='line':
            a1, = plt.plot(x_, y, linestyle, c=color, linewidth=linewidth,
                           ms=10, label=label)
        elif style=='bar':
            a1 = plt.bar(x_, y,color=color,label=label,width=width)
        if 'RSDttest' in key_:
            cc = dic_y_['RSDcolor']
            T = dic_y_['RSDttest']
            breakpoints = RSDttest(y, T)
            breakpoints.insert(0, 0)
            breakpoints.append(len(y)-1)
            for j in range(len(breakpoints)-1):
                start = breakpoints[j]
                end = breakpoints[j+1]
                p = numpy.polyfit(x_[start:end+1], y[start:end+1], 1)
                y_ = numpy.polyval(p, x_[start:end+1])
                plt.plot(x_[start:end+1], y_, c=cc, linewidth=4)
        if 'trend' in key_:
            cc = dic_y_['trend']
            p = numpy.polyfit(x_, y, 1)
            y_ = numpy.polyval(p, x_)
            r,pp = scipy.stats.pearsonr(x_,y)
            if pp>0.1:
                cp = cc[0]
            elif 0.1>=pp>0.05:
                cp = cc[1]
            elif pp<0.05:
                cp = cc[2]
            plt.plot(x_, y_, c=cp, linewidth=4)
        if not (label is None):
            A.append(a1)
    plt.xlim(xt[0], xt[-1])
    if y_lim:
        plt.ylim(y_lim[0], y_lim[1])
    if str_x:
        plt.xticks(xt, x)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if rotation=='auto':
        fig1.autofmt_xdate()
    else:
        for tick in ax.get_xticklabels():
            tick.set_rotation(int(rotation))
    plt.xlabel(xlabel, fontsize=fz)
    plt.ylabel(ylabel, fontsize=fz)
    # ax.grid(color='gray', linestyle='-.', linewidth=1)
    if make_legend:
        plt.legend(handles=A, loc=loc,
                   bbox_to_anchor=(1, 1), fontsize=fz)
    ax.tick_params(which='major', width=4, length=20)
    ax.tick_params(labelsize=fz)
    if minor:
        ax.minorticks_on()
        ax.tick_params(which='minor',width=4,length=10)
    if save:
        plt.savefig(name, dpi=dpi, bbox_inches='tight')
    plt.close(fig1)


def paint_bar(x, dic_y, xlabel, ylabel, name,
              dpi=300, save=True, make_legend=False):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fz = 40
    fig1 = plt.figure(1, figsize=(18, 12))
    ax = fig1.add_subplot(111)
    A = []
    keys = numpy.array(list(dic_y.keys()))
    nn = len(keys)
    xt = numpy.arange(len(x))
    for ikey in range(nn):
        key = keys[ikey]
        dic_y_ = dic_y[key]
        y = dic_y_['y']
        color = dic_y_['color']
        label = None
        if 'label' in list(dic_y_.keys()):
            label = dic_y_['label']
        a1 = plt.bar(xt, y, color=color, label=label)
        A.append(a1)
    plt.xticks(xt, x)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    fig1.autofmt_xdate()
    plt.xlabel(xlabel, fontsize=fz)
    plt.ylabel(ylabel, fontsize=fz)
    if numpy.min(y) < 0:
        std = max(abs(y))
        plt.ylim(-1*std, std)
    if make_legend:
        plt.legend(handles=A, loc='center right',
                   bbox_to_anchor=(1, 0.9), fontsize=fz)
    ax.tick_params(labelsize=fz)
    # ax.grid(color='k', linestyle='--', linewidth=1, axis='y')
    if save:
        plt.savefig(name, dpi=dpi, bbox_inches='tight')
    plt.close(fig1)


def link_figure(pic_list, name, dic_font=None, pad=0, q=100):
    # pic_list[[name1,name2,...,namenx],...*ny]
    # dic_font = {'0':{'size':...,'position':(percent,percent),
    # 'color':...,'text':...,}}
    from PIL import Image, ImageDraw, ImageFont
    Image.MAX_IMAGE_PIXELS = 200000000
    temp_target = []
    ny = len(pic_list)
    nx = len(pic_list[0])
    ws_ = []
    hs_ = []
    for iy in range(ny):
        pic_list_ = pic_list[iy]
        ws = []
        hs = []
        for ix in range(nx):
            im = Image.open(pic_list_[ix])
            w, h = im.size
            ws.append(w)
            hs.append(h)
        ws = [int(ws[ix]/hs[ix]*hs[0]) for ix in range(nx)]
        hs = [hs[0]]*nx
        target = Image.new('RGB', (sum(ws), hs[0]), color='white')
        left, up = 0, 0
        right, down = ws[0], hs[0]
        for ix in range(nx):
            im = Image.open(pic_list_[ix])
            im = im.resize((ws[ix], hs[ix]))
            # print(im.size, left, up, right, down)
            target.paste(im, (left, up, right, down))
            if ix < nx-1:
                left += ws[ix]
                right += ws[ix+1]
        temp_target.append(target)
        ws_.append(sum(ws))
        hs_.append(hs[0])
    hs_ = [int(hs_[iy]/ws_[iy]*ws_[0]) for iy in range(ny)]
    ws_ = [ws_[0]]*ny
    pad_h = int(pad*hs_[0])
    target_final = Image.new('RGB', (ws_[0], sum(hs_)+(ny)*pad_h),
                             color='white')
    left, up = 0, pad_h
    right, down = ws_[0], hs_[0]+pad_h
    for iy in range(ny):
        im = temp_target[iy]
        im = im.resize((ws_[iy], hs_[iy]))
        target_final.paste(im, (left, up, right, down))
        if iy < ny-1:
            up += (hs_[iy]+pad_h)
            down += (hs_[iy+1]+pad_h)
    w, h = target_final.size
    if dic_font is None:
        pass
    else:
        draw = ImageDraw.Draw(target_final)
        keys = list(dic_font.keys())
        for key in keys:
            position = list(dic_font[key]['position'])
            text = dic_font[key]['text']
            color = dic_font[key]['color']
            size = dic_font[key]['size']
            ft = ImageFont.truetype(font=r'C:\Windows\Fonts\simsun.ttc',
                                    size=size)
            position[0] *= w
            position[1] *= h
            position = (position[0], position[1])
            draw.text(position, text, font=ft, fill=color)
    target_final.save(name, quality=q)


def extend_figure(pic, name,
                  left=0, right=0, top=0, bottom=0, fill_color='white'):
    from PIL import Image
    im = Image.open(pic)
    w, h = im.size
    left_extend = int(w*left)
    right_extend = int(w*right)
    top_extend = int(h*top)
    bottom_extend = int(h*bottom)
    target = Image.new('RGB',
                       (w+left_extend+right_extend,
                        h+top_extend+bottom_extend),
                       color='white')
    target.paste(im, (left_extend, top_extend, left_extend+w, top_extend+h))
    target.save(name, quality=100)


def eof(data, standard=False):  # (data,time)
    import scipy.linalg
    import numpy
    import math
    ndata = data.shape[0]
    ntime = data.shape[1]
    time = list(range(0, ntime))
    aa = numpy.ones((ndata, ntime)) - 1
    mean_a = numpy.mean(data, axis=1)
    for i in time:
        aa[:, i] = data[:, i] - mean_a
    if standard:
        for i in range(ndata):
            if numpy.std(aa[i, :])!=0:
                aa[i, :]/=numpy.std(aa[i, :])
    u, s, vh = scipy.linalg.svd(aa)
    # different from matlab, in python a = u*ss*vh;while in mtlab, a = u*ss*vh'
    # print(u)
    # print(s)
    # print(vh)
    eof = u  # eof[:,0] is mode1;[:,1] is mode2 and so on
    x = len(s)
    ss = numpy.ones(data.shape) - 1
    for i in range(0, x):
        ss[i, i] = s[i]
    ee = ss ** 2 / ntime
    pc = numpy.dot(ss, vh)
    # pc[0,:] is mode1;[1,:] is mode2 and so on
    num = numpy.ones([x]) - 1
    for i in range(0, x):
        num[i] = ee[i, i]
    percent = num / sum(num)
    # north test
    north = -1
    derta = numpy.ones([x]) - 1
    for count in range(0, x):
        derta[count] = percent[count] * (math.pow(2 / (ntime - 1), 0.5))
    for count in range(0, x - 1):
        if ((percent[count] - derta[count]) < (
                percent[count + 1] + derta[count + 1])):
            north = count
            break
    # print(percent,derta)
    for count in range(x):
        de = numpy.std(pc[count, :])
        # max(abs(min(pc[count,:])),abs(max(pc[count,:])))#math.std(pc[count,:])
        eof[:, count] = eof[:, count] * de
        pc[count, :] = pc[count, :] / de
    return eof, pc, percent, north


def seof(sdata, do_aver=True, do_std=False, cov=False, corr=False):
    # (season,data,time)
    import scipy.linalg
    import numpy
    import math
    from demo_function import mean_data
    if do_std:
        do_aver = False
    if cov:
        corr = False
    if corr:
        cov = False
    nseason = sdata.shape[0]
    nsdata = sdata.shape[1]
    ntime = sdata.shape[2]
    time = list(range(0, ntime))
    data = sdata.reshape([-1, ntime])
    aa = numpy.ones(data.shape) - 1
    if do_aver:
        ndata = nseason * nsdata
        option = {
            'order': ['data', 'time'],
            'standard': 'time',
            'range': time,
        }
        mean_a = mean_data(data, option)
        for i in time:
            aa[:, i] = data[:, i] - mean_a
    elif do_std:
        ndata = nseason * nsdata
        option = {
            'order': ['data', 'time'],
            'standard': 'time',
            'range': time,
        }
        mean_a = mean_data(data, option)
        for i in time:
            aa[:, i] = data[:, i] - mean_a
            aa[:, i] /= numpy.std(aa[:, i])
    elif not do_aver and not do_std:
        aa = data
    u, s, vh = scipy.linalg.svd(aa)
    # different from matlab, in python a = u*ss*vh;while in mtlab, a = u*ss*vh'
    # print(u)
    # print(s)
    # print(vh)
    eof = u  # eof[:,0] is mode1;[:,1] is mode2 and so on
    x = len(s)
    ss = numpy.ones(data.shape) - 1
    for i in range(0, x):
        ss[i, i] = s[i]
    ee = ss ** 2 / ntime
    pc = numpy.dot(ss, vh)
    # pc[0,:] is mode1;[1,:] is mode2 and so on
    num = numpy.ones([x]) - 1
    for i in range(0, x):
        num[i] = ee[i, i]
    percent = num / sum(num)
    # north test
    north = -1
    derta = numpy.ones([x]) - 1
    for count in range(0, x):
        derta[count] = percent[count] * (math.pow(2 / (ntime - 1), 0.5))
    for count in range(0, x - 1):
        if ((percent[count] - derta[count]) < (
                percent[count + 1] + derta[count + 1])):
            north = count
            break
    # print(percent,derta)
    if (not cov) and (not corr):
        for count in range(x):
            de = s[count]
            # max(abs(min(pc[count,:])),abs(max(pc[count,:])))#math.std(pc[count,:])
            eof[:, count] = eof[:, count] * de
            pc[count, :] = pc[count, :] / de
        seof = eof[:, 0:x].reshape([nseason, nsdata, x])
    elif cov:
        cov_eof = numpy.ones(eof.shape) - 1
        for count in range(x):
            pc[count, :] -= numpy.mean(pc[count, :])
            pc[count, :] /= numpy.std(pc[count, :])
            for i in range(eof.shape[0]):
                cov = numpy.cov(pc[count, :], aa[i, :])[0, 1]
                if not math.isnan(cov):
                    cov_eof[i, count] = cov
        seof = cov_eof[:, 0:x].reshape([nseason, nsdata, x])
    elif corr:
        cov_eof = numpy.ones(eof.shape) - 1
        for count in range(x):
            pc[count, :] -= numpy.mean(pc[count, :])
            pc[count, :] /= numpy.std(pc[count, :])
            for i in range(eof.shape[0]):
                cov = numpy.corrcoef(pc[count, :], aa[i, :])[0, 1]
                # print(cov)
                if not math.isnan(cov):
                    cov_eof[i, count] = cov
        seof = cov_eof[:, 0:x].reshape([nseason, nsdata, x])
    return seof, pc[0:x, :], percent, north



#@numba.jit(nopython=True)
def spherical_distance(point1, point2, R=6371):
    # point should be input as [ lat, lon ] ,
    # positive for North , negative for South
    # Pi = 3.1415926
    dlat = abs(point1[0] - point2[0]) / 180 * Pi
    dlon = abs(point1[1] - point2[1]) / 180 * Pi
    lat1 = point1[0] / 180 * Pi
    lat2 = point2[0] / 180 * Pi
    l_p1_p12 = 2 * numpy.sin(dlon / 2) * numpy.cos(lat1)
    l_p1_p21 = 2 * numpy.sin(dlat / 2)
    l_p2_p21 = 2 * numpy.sin(dlon / 2) * numpy.cos(lat2)
    l_p1_p2 = numpy.sqrt(l_p1_p21 ** 2 + l_p2_p21 * l_p1_p12)
    theta = 2 * numpy.arcsin(1 / 2 * l_p1_p2)
    dis = theta * R
    return dis


def spherical_ang(A, B, C):
    # A, B, C is three point with order, A to B to C
    # cosb=cosccosa+sincsinacosB
    # return B
    import numpy
    R = 1
    a = spherical_distance(B, C, R=R)
    b = spherical_distance(A, C, R=R)
    c = spherical_distance(A, B, R=R)
    cosb = numpy.cos(b)
    cosc = numpy.cos(c)
    cosa = numpy.cos(a)
    sinc = numpy.sin(c)
    sina = numpy.sin(a)
    if sina==0 or sinc==0:
        theta = 0
    elif sina!=0 and sinc!=0:
        cosB = (cosb-cosc*cosa)/(sinc*sina)
        if numpy.absolute(cosB)>1:
            cosB = cosB/numpy.absolute(cosB)
        #print(cosB)
        theta = numpy.arccos(cosB)
    return theta

@numba.jit(nopython=True)
def make_global_extend(varxy,lon,lat,derta=1,vector=True,north=True,south=True):
    # lon: 0-359.5
    # lat: 90--90

    Nj = varxy.shape[0]
    Ni = varxy.shape[1]
    lon_extend = numpy.ones((Nj+2*derta,Ni+2*derta))-1
    #print(lon_extend.shape)
    lat_extend = numpy.ones((Nj+2*derta,Ni+2*derta))-1

    varreg_extend = numpy.ones((Nj+2*derta,Ni+2*derta))-1
    lon_limit = lon[0, :]
    lat_limit = lat[:, 0]
    # c
    c = numpy.ones(lon_extend.shape)
    if vector :
        c[0:derta, :]=-1
        c[Nj+derta:Nj+2*derta, :]=-1
    # lon
    lon_extend[0:derta, derta:Ni+derta] = lon[1:derta+1, ::][::-1, :]+180
    lon_extend[Nj+derta:Nj+2*derta, derta:Ni+derta] =  lon[-1*derta-1:-1, ::][::-1, :]+180
    lon_extend[derta:Nj+derta, derta:Ni+derta] = lon.copy()
    lon_extend[:, Ni+derta:Ni+2*derta] = lon_extend[:, derta:2*derta]
    lon_extend[:, 0:derta] = lon_extend[:,-2*derta-1:-1*derta-1]
    for index, ndata in numpy.ndenumerate(lon_extend):
        if ndata>=360:
            lon_extend[index]-=360
    # lat
    lat_extend[0:derta, derta:Ni+derta] = lat[1:derta+1, ::][::-1, :]
    lat_extend[Nj+derta:Nj+2*derta, derta:Ni+derta] =  lat[-1*derta-1:-1, ::][::-1, :]
    lat_extend[derta:Nj+derta, derta:Ni+derta] = lat.copy()
    lat_extend[:, Ni+derta:Ni+2*derta] = lat_extend[:, derta:2*derta]
    lat_extend[:, 0:derta] = lat_extend[:,-2*derta:-1*derta]
    for index, ndata in numpy.ndenumerate(varreg_extend):
        # print(lat_extend[index], lon_extend[index])
        iy = numpy.where(lat_limit==lat_extend[index])[0][0]
        ix = numpy.where(lon_limit==lon_extend[index])[0][0]
        varreg_extend[index] = varxy[iy, ix]
    varreg_extend*=c

    if north and south :
        vv = varreg_extend.copy()
        lo = lon_extend.copy()
        la = lat_extend.copy()
    elif (not north) and (not south) :
        vv = varreg_extend[derta:-1*(derta),:]
        lo = lon_extend[derta:-1*(derta),:]
        la = lat_extend[derta:-1*(derta),:]
    elif (north) and (not south) :
        vv = varreg_extend[0:-1*(derta),:]
        lo = lon_extend[0:-1*(derta),:]
        la = lat_extend[0:-1*(derta),:]
    elif (not north) and (south) :
        vv = varreg_extend[derta::,:]
        lo = lon_extend[derta::,:]
        la = lat_extend[derta::,:]
    return vv,lo,la


def get_ticks(x):
    x_ticks = [x[i]+'-'+x[i+1] for i in range(len(x)-1)]
    x_ticks.insert(0, '<'+x[0])
    x_ticks.append('>'+x[-1])
    return x_ticks


def multi_intersection(arr1, arr2):
    # 两个数组取交集
    arr1_view = arr1.view([('', arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('', arr2.dtype)]*arr2.shape[1])
    intersected = numpy.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

def iner_point(vertex_list, x_range, y_range):
    # vertex:(x,y)
    # vertex_list should be ordered!
    # x_range and y_range should be linear
    import numpy
    def intersection(A, B):
        answer = list(set(A).intersection(set(B)))
        return answer


    def iner(vertex_list, point):
        # return 1 means iner
        # return 0 means outer
        # !!linking a point and vertexs directionally,if the point is an iner point ,the sum of the direction angles(whoes abs < 180)
        # !!should be (2k+1)*2Pi ,while it should be (2k)*2Pi for an outer point , where k belongs to K+.
        n = len(vertex_list)
        vector = numpy.ones([n + 1], dtype='complex') - 1
        ang = numpy.ones([n]) - 1
        for i in range(n):
            vector[i] = (-point[0] + vertex_list[i][0]) + (-point[1] + vertex_list[i][1]) * 1j
            vector[i] /= abs(vector[i])
        vector[-1] = vector[0]
        for i in range(n):
            cc = (vector[i + 1] * vector[i].conjugate()).real
            sig = (vector[i + 1] / vector[i]).imag
            if sig >= 0:
                sig = 1
            elif sig < 0:
                sig = -1
            if cc >= 0:
                cc = min(cc, 1)
            elif cc < 0:
                cc = max(-1, cc)
            ang[i] = numpy.arccos(cc) * sig
        return int(sum(ang) / 6.28) % 2
    vertex_list = list(vertex_list)
    vertex_x_list = [vertex[0] for vertex in vertex_list]
    vertex_y_list = [vertex[1] for vertex in vertex_list]
    x_min = min(vertex_x_list)
    x_max = max(vertex_x_list)
    y_min = min(vertex_y_list)
    y_max = max(vertex_y_list)
    x_range = numpy.array(x_range)
    y_range = numpy.array(y_range)
    x_lim_1 = list(x_range[numpy.where((x_range >= x_min))])
    y_lim_1 = list(y_range[numpy.where((y_range >= y_min))])
    x_lim_2 = list(x_range[numpy.where((x_range <= x_max))])
    y_lim_2 = list(y_range[numpy.where((y_range <= y_max))])
    x_lim = intersection(x_lim_1, x_lim_2)
    y_lim = intersection(y_lim_1, y_lim_2)
    point_temp = []
    for xx in x_lim:
        for yy in y_lim:
            if (xx, yy) not in vertex_list:
                # print([xx,yy])
                point_temp.append((xx, yy))
    # print(x_min,x_max,y_min,y_max,x_lim,y_lim)
    iner_points = []
    for point in point_temp:
        # print(point)
        if iner(vertex_list, point) == 1:
            iner_points.append(point)
    iner_points += vertex_list
    return iner_points


def normalize(y):
    import numpy
    ans = numpy.array(y)
    ans[:]-=numpy.min(ans[:])
    d = numpy.max(ans)-numpy.min(ans)
    if d!=0:
        ans/=d
    return ans


def detrend(y):
    import numpy
    x = numpy.arange(len(y))
    p = numpy.polyfit(x,y,1)
    y_ = numpy.polyval(p, x)
    x_ = y - y_
    return x_


def move_average(y, delay, omega=None):
    import numpy
    if omega is None:
        omega = numpy.ones((2*delay+1))
    if len(y)<2*delay+1:
        print('wrong,y is too short')
        return None
    elif len(y)>= 2*delay+1:
        ans = numpy.ones((len(y)-2*delay))-1
        for i in range(len(ans)):
            ans[i] = numpy.mean(y[i:2*delay+1+i]*omega)
        return ans


def remove_average(y, delay, omega=None):
    ans = y[delay:-1*delay]-move_average(y, delay, omega=omega)
    return ans