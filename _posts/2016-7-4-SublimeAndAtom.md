---
layout: post
title: 常用Sublime和Atom插件
---
### Sublime插件

先装插件管理器(Package Control)：https://packagecontrol.io/installation

```
import urllib.request,os,hashlib; h = '2915d1851351e5ee549c20394736b442' + '8bc59f460fa1548d1514676163dafc88'; pf = 'Package Control.sublime-package'; ipp = sublime.installed_packages_path(); urllib.request.install_opener( urllib.request.build_opener( urllib.request.ProxyHandler()) ); by = urllib.request.urlopen( 'http://packagecontrol.io/' + pf.replace(' ', '%20')).read(); dh = hashlib.sha256(by).hexdigest(); print('Error validating download (got %s instead of %s), please try manual install' % (dh, h)) if dh != h else open(os.path.join( ipp, pf), 'wb' ).write(by)
```

- **ConvertToUTF8** : GBK编码兼容。
- **[wakatime](https://wakatime.com/)** :自动记录code时间，支持多种编辑器和IDE。
先到官网注册，登录后在右上角点用户名，选择Setting，左侧选Account，复制Api Key。Sublime中安装此插件会用到。以后就可以登录网站查看自己的code时间统计图。
- **SpaceGray** : 最喜欢的主题
- **SublimeAStyleFormatter** : 自动对齐
- **BracketHilighter**：括号高亮
- **C++NamespaceTool**
- **SublimeLinter**
- **FileDiffs**
- **Auto Semi-Colon**：TAB自动跳出括号


### Atom插件

- **markdown-preview-enhanced** ：很好用的Markdown编辑器。
- **Autocomplete-paths** ：各种Autocomplete系列。
- **python-tools**
- **highlight_selected**：
- **minimap**：类似于sublime
