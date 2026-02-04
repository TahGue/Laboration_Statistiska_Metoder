# Laboration - Statistiska Metoder (Bostadspriser)

Här är min lösning på laborationen i statistiska metoder. Målet var att bygga en **Multipel Linjär Regression (OLS)** helt från grunden utan att använda färdiga ML-bibliotek som sklearn, utan bara ren matte med `numpy` och `scipy.stats`.

## 📂 Filer i projektet

*   `linear_regression.py`: Här ligger all logik och min klass `LinearRegression`. Den sköter själva uträkningarna (matrisberäkningar, t-tester, F-test osv).
*   `lab.ipynb`: Jupyter Notebooken som laddar datan, kör modellen och visar upp resultaten och analysen.
*   `housing.csv`: Datasetet (inte inkluderat i git-repot, men behövs för att köra koden).

## � Om lösningen (VG-krav)

Jag har siktat på att uppfylla kraven för **Väl Godkänt (VG)** genom att implementera följande:

1.  **Numerisk stabilitet**: Eftersom datan har variabler som hänger ihop mycket (hög korrelation) så använder jag Moore-Penrose pseudo-invers (`pinv`) istället för vanlig invers. Det gör att det inte kraschar när man slänger in alla variabler.
2.  **Fullständig statistik**:
    *   Räknar ut koefficienter, standardfel och t-värden för varje variabel.
    *   Konfidensintervall (går att ställa in nivå själv, t.ex. 95% eller 99%).
    *   F-test för att se om hela modellen är signifikant.
    *   Pearson-korrelation för **alla** par av variabler (visas som en matris i notebooken).
3.  **Kategorisk data**: Fixat så att `ocean_proximity` görs om till siffror automatiskt (One-Hot Encoding).
4.  **Presentation**: Har lagt in en klass (`RegressionResults`) som gör att man får en snygg tabell i Notebooken utan att bryta mot regeln om "inga print-satser i cellerna".

## ▶️ Så här kör du koden

1.  Se till att du har `housing.csv` i samma mapp.
2.  Dra igång Jupiter:
    ```bash
    jupyter notebook
    ```
3.  Öppna `lab.ipynb` och kör alla celler uppifrån och ner.

## 📝 Notering

Jag har valt att ta med **alla** variabler i modellen. Det finns en del multikollinearitet (särskilt mellan antal rum, sovrum och befolkning), men jag har låtit det vara kvar för att kunna analysera det statistiskt i notebooken, vilket jag diskuterar där.
