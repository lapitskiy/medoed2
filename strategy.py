import emoji #https://carpedm20.github.io/emoji/

stg_dict = {
    'ladder_stg':
        {
            'name': 'Лесенка',
            'desc': 'Стратегия торговли лесенкой',
            'step': '0,0',
            'amount': 0
        }}

def runStg(stg_name: str):
    if stg_name == 'ladder_stg':
        createStg = Strategy_Step()
        return createStg.returnHelpInfo()

class Strategy_Step():
    step: float

    def returnStgDict(self, step: str) -> dict:
        self.step = float(step)
        ddict = {
            'step': step,
        }
        return ddict

    def returnHelpInfo(self) -> str:
        return f"Настройка стратегии Лесенка" + emoji.emojize(":chart_increasing:") \
               + "\nВводите команды для настройки стратегии\n" \
               + "\n/stgedit ladder_stg active True - Выбрать эту стратегию\n" \
               +"/stgedit ladder_stg step 0.5 - шаг в USDT\n" \
                "/stgedit ladder_stg amount 100 - сколько USDT за одну сделку\n"




