# Passive Aggressive
![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)

![ezgif com-resize](https://user-images.githubusercontent.com/26996041/39611189-d019a09c-4f90-11e8-853c-75d6ef24bc02.gif)
</br>  
**This python script visualizes the process of Online Passive-Aggressive (PA) algorithm and PA-Ⅰ algorithm.**
</br>   
PA algorithm is a margin based online learning algorithm[Crammer et al.] for binary classification.  
Unlike PA algorithm, which is a hard-margin based method, PA-I algorithm is a soft margin based method and robuster to noise.

>"Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz, Yoram Singer, "Online Passive-Aggressive Algorithms" Journal of Machine Learning Research 7 (2006) 551–585."

#### 日本語解説ページ (Qiita)
[オンライン機械学習の弱点って？？：Passive Aggressiveのプロセスを実装＆可視化](https://qiita.com/Wotipati/items/a8eda3f246eb07c516ca)

## Example
#### Show online learning process:
```
python passive_aggressive_ex.py
```


#### If you want to record the result:
```
python passive_aggresive_ex.py --record 1
```
The result is saved as "results.mp4"

---

### References
- [オンライン機械学習 (機械学習プロフェッショナルシリーズ)](https://www.kspub.co.jp/book/detail/1529038.html)
- [統計の素人だけどPythonで機械学習モデルを実装したい、そんな人のための第一歩](https://qiita.com/hik0107/items/9b6e1e989f4eaefdc31d)
- [実装して理解するオンライン学習器(1) - PassiveAggresive](http://smrmkt.hatenablog.jp/entry/2014/10/13/124757)
- [AtsushiSakai/matplotrecorder](https://github.com/AtsushiSakai/matplotrecorder)

### License
MIT License, Seita Kayukawa (Wotipati)
