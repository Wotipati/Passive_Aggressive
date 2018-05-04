# Passive Aggressive

![ezgif com-resize](https://user-images.githubusercontent.com/26996041/39611189-d019a09c-4f90-11e8-853c-75d6ef24bc02.gif)


**This python script visualizes the process of Online Passive-Aggressive (PA) algorithm and PA-Ⅰ algorithm.**   


PA algorithm is a margin based online learning algorithm[Crammer et al.] for binary classification.  
Unlike PA algorithm which is a hard-margin based method, PA-Ⅰ algorithm is a soft margin based method and robuster to noise.

>"Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz, Yoram Singer, "Online Passive-Aggressive Algorithms" Journal of Machine Learning Research 7 (2006) 551–585."



## Example
#### Show online learning process:
```
python passive_aggressive_ex.py
```


#### If you want to record results:
```
python passive_aggresive_ex.py --record 1
```
The results is saved as "results.mp4"

---

### References
- [オンライン機械学習 (機械学習プロフェッショナルシリーズ)](https://www.kspub.co.jp/book/detail/1529038.html)
- [統計の素人だけどPythonで機械学習モデルを実装したい、そんな人のための第一歩](https://qiita.com/hik0107/items/9b6e1e989f4eaefdc31d)
- [実装して理解するオンライン学習器(1) - PassiveAggresive](http://smrmkt.hatenablog.jp/entry/2014/10/13/124757)
- [AtsushiSakai/matplotrecorder](https://github.com/AtsushiSakai/matplotrecorder)
