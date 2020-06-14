def elev(latmin , latmax , lonmin , lonmax , * , resolution_lat = 1 , resolution_lon = 1 , mode = 180):
    #pay attention!!in this dataset,x is from -180 to 180,y is from -90 to 90!!
    import numpy
    import netCDF4 as nc
    from demo_function import interp2
    import gc
    path = r'C:\data\elev'
    name = 'ETOPO1_Ice_g_gmt4.grd'
    ds   = nc.Dataset(path+'\\'+name)
    #x    = ds.variables['x'][:]
    #y    = ds.variables['y'][:]
    dx   = 360/21600
    if lonmax >= 180 and mode == 180 :
        print('Warning : if longitude is set from 0(0E) to 360(0W) , please input "mode = 360"')
    if lonmin < 0 and mode == 360 :
        print('Warning : if longitude is set from -180(180W) to 180(180E) , please input "mode = 180"')
    lat_alter  = numpy.arange(-90,90+resolution_lat,resolution_lat)
    if mode == 180 :
        lon0 = numpy.arange(-180,180+resolution_lon,resolution_lon)
        if lonmin <-180 or lonmax >180 :
            extend = True
            x_extend_left = numpy.arange(-180-dx,lonmin,-1*dx)[::-1]
            x_extend_right= numpy.arange(180+dx,lonmax,dx)
            x_extend_left_new = numpy.arange(-180-resolution_lon,lonmin,-1*resolution_lon)[::-1]
            x_extend_right_new= numpy.arange(180+resolution_lon,lonmax,resolution_lon)
            extend_left = len(x_extend_left)
            extend_right= len(x_extend_right)
        else :
            extend = False
        z    = ds.variables['z'][:]

    elif mode == 360 :
        lon0 = numpy.arange(0,360+resolution_lon,resolution_lon)
        if lonmin < 0 or lonmax > 360 :
            extend = True
            x_extend_left = numpy.arange(0-dx,lonmin,-1*dx)[::-1]
            x_extend_right= numpy.arange(360+dx,lonmax,dx)
            x_extend_left_new = numpy.arange(0-resolution_lon,lonmin,-1*resolution_lon)[::-1]
            x_extend_right_new= numpy.arange(360+resolution_lon,lonmax,resolution_lon)
            extend_left = len(x_extend_left)
            extend_right= len(x_extend_right)
        else :
            extend = False
        z_temp = ds.variables['z'][:]
        z      = numpy.ones([10801,21601])-1
        z[:,0:10801] = z_temp[:,10800::].copy()
        z[:,10801::] = z_temp[:,0:10800].copy()
        del z_temp
        gc.collect()
    if extend :
        lon_alter = numpy.append(numpy.append(x_extend_left_new,lon0),x_extend_right_new)
    elif not extend :
        lon_alter = lon0
        extend_left = 0
        extend_right= 0
    zz        = numpy.ones([10801,21601+extend_left+extend_right])
    zz[:,0:extend_left] = z[:,21601-1*extend_left::]
    zz[:,extend_left:21601+extend_left] = z[:,:]
    zz[:,21601+extend_left::] = z[:,0:extend_right]
    del z
    gc.collect()
    shape_alter = (len(lat_alter),len(lon_alter))
    z_alter   = interp2(zz,shape_alter)
    x_start   = numpy.where(abs(lon_alter - lonmin) < resolution_lon)[0][0]
    x_end     = numpy.where(abs(lonmax - lon_alter) < resolution_lon)[0][0]
    y_start   = numpy.where(abs(lat_alter - latmin) < resolution_lat)[0][0]
    y_end     = numpy.where(abs(latmax - lat_alter) < resolution_lat)[0][0]
    if lon_alter[x_start] < lonmin :
        x_start = x_start + 1
    if lat_alter[y_start] < latmin :
        y_start = y_start + 1
    data      = z_alter[y_start:y_end+1,x_start:x_end+1]
    x_alter   = lon_alter[x_start:x_end+1]
    y_alter   = lat_alter[y_start:y_end+1]
    del z_alter
    gc.collect()
    return data,y_alter,x_alter

def main():
    import numpy
    from background import plot_data
    from demo_function import getcolorbar
    extend = False
    lon1               = -100
    lon2               = 330
    lat1               = -90
    lat2               = 90
    print(lon1,lon2,lat1,lat2)
    kk,lat_alter,lon_alter=elev(lat1,lat2,lon1,lon2,resolution_lon = 0.1,resolution_lat = 0.1 , mode =360)
    basemap_option = {
        'llcrnrlon'  : lon1,
        'llcrnrlat'  : lat1,
        'urcrnrlon'  : lon2,
        'urcrnrlat'  : lat2,
        'projection' : 'cyl',
        }
    fig_size = (24,16)
    globe = True
    adress= [0,0,0,1]
    colormap = getcolorbar('GMT_globe')#NCV_gebco
    plot_option = {
    'adress'         : adress,
    'basemap_option' : basemap_option,
    'coast'          : True,
    'colormap'       : colormap,
    'dtype'          : 'float',
    'element'        : 'elev',
    'fig_size'       : fig_size,
    'file_dir'       : 'elevation_0.013deg',
    'lat'            : lat_alter,
    'level'          : False,
    'levels'         : numpy.arange(-11000,11000,10),
    'lon'            : lon_alter,
    'mask'           : False,
    'mask1'          : None,
    'save'           : False,
    'stream'         : False,
    'type_data'      : 'global_0.5deg',
    'unit'           : 'm',
    'line'           : False,
    'show'           : True,
    }
    print(numpy.min(kk),numpy.max(kk))
    plot_option['data'] = kk
    plot_option['backdata'] = None
    plot_option['title'] = 'test'
    plot_data(**plot_option)

if __name__ == '__main__' :
    main()
