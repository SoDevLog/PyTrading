import os
from styles import palette_colors
from matplotlib.font_manager import FontProperties
        
class Watermark:

    font_filename_path = os.path.dirname( __file__ )
    font_filename_path = os.path.join( font_filename_path, "Montserrat-ExtraBold.ttf" )
    
    # print( f"Font file path: {font_filename_path}" )
    
    font = FontProperties(
        family="Montserrat",
        fname=font_filename_path,
        weight="600"
    )

    text = "TradingInPython"
    fontsize = 38
    rotation = 0
    alpha = 0.1
    x = 0.5
    y = 0.11

    @classmethod
    def apply(cls, fig):

        return fig.text(
            cls.x,
            cls.y,
            cls.text,
            transform=fig.transFigure,
            ha="center",
            va="center",
            fontproperties=cls.font,
            rotation=cls.rotation,
            fontsize=cls.fontsize,
            color=palette_colors.WATERMARK,
            alpha=cls.alpha,
            zorder=1000,
        )