import re

import gradio as gr
import modules.scripts as scripts
from modules import script_callbacks
from modules import generation_parameters_copypaste as params_copypaste
from modules.processing import (
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
)
from modules.scripts import basedir, OnComponent
from modules.shared import opts

import scripts.t2p.settings as settings

if settings.DEVELOP:
    import scripts.t2p.prompt_generator as pgen
    from scripts.t2p.prompt_generator.wd_like import WDLike
else:
    from scripts.t2p.dynamic_import import dynamic_import
    _wd_like = dynamic_import('scripts/t2p/prompt_generator/wd_like.py')
    WDLike = _wd_like.WDLike
    pgen = _wd_like.pgen

wd_like = WDLike()

# brought from modules/deepbooru.py
re_special = re.compile(r'([\\()])')

def get_conversion(choice: int):
    if choice == 0: return pgen.ProbabilityConversion.CUTOFF_AND_POWER
    elif choice == 1: return pgen.ProbabilityConversion.SOFTMAX
    else: raise NotImplementedError()

def get_sampling(choice: int):
    if choice == 0: return pgen.SamplingMethod.NONE
    elif choice == 1: return pgen.SamplingMethod.TOP_K
    elif choice == 2: return pgen.SamplingMethod.TOP_P
    else: raise NotImplementedError()

def get_tag_range_txt(tag_range: int):
    if wd_like.database is None:
        return 'Tag range: NONE'
    maxval = len(wd_like.database.tag_idx) - 1
    i = max(0, min(tag_range, maxval))
    r = wd_like.database.tag_idx[i]
    return f'Tag range: <b> ≥ {r[0]} tagged</b> ({r[1] + 1} tags total)'

def dd_database_changed(database_name: str, tag_range: int):
    wd_like.load_data(database_name)
    return [
        gr.Slider.update(tag_range, 0, len(wd_like.database.tag_idx) - 1),
        get_tag_range_txt(tag_range)
    ]

def sl_tag_range_changed(tag_range: int):
    return get_tag_range_txt(tag_range)

def generate_prompt(text: str, text_neg: str, neg_weight: float, tag_range: int, conversion: int, power: float, sampling: int, n: int, k: int, p: float, weighted: bool, replace_underscore: bool, excape_brackets: bool):
    wd_like.load_model() #skip loading if not needed
    tags = wd_like(text, text_neg, neg_weight, pgen.GenerationSettings(tag_range, get_conversion(conversion), power, get_sampling(sampling), n, k, p, weighted))
    if replace_underscore: tags = [t.replace('_', ' ') for t in tags]
    if excape_brackets: tags = [re.sub(re_special, r'\\\1', t) for t in tags]
    return ', '.join(tags)


class Text2PromptScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.prompt_area = [None, None]  # txt2img, img2img
        self.text2prompt_areas = [None, None]  # 存储我们创建的UI组件
        self.on_after_component_elem_id = [
            ("txt2img_prompt_row", lambda x: self.create_text2prompt_area(0, x)),
            ("txt2img_prompt", lambda x: self.set_prompt_area(0, x)),
            ("img2img_prompt_row", lambda x: self.create_text2prompt_area(1, x)),
            ("img2img_prompt", lambda x: self.set_prompt_area(1, x)),
        ]

    def title(self):
        return "Text2Prompt"

    def show(self, _):
        return scripts.AlwaysVisible

    def create_text2prompt_area(self, i2i: int, prompt_row: OnComponent):
        """在主提示词下方创建Text2Prompt输入区域"""
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<h3>🎨 Text2Prompt 智能提示词生成</h3>')
                tb_input = gr.Textbox(
                    label='主题描述',
                    interactive=True,
                    placeholder='输入你想要生成的内容主题，例如：蓝色的头发，白色的连衣裙',
                    lines=2
                )
                tb_input_neg = gr.Textbox(
                    label='负面主题',
                    interactive=True,
                    placeholder='不希望出现的内容，例如：低质量，模糊',
                    lines=2
                )
                with gr.Row():
                    btn_generate = gr.Button(value='🚀 生成提示词', variant='primary', size='lg')
                    btn_clear = gr.Button(value='🗑️ 清空', variant='secondary')

            with gr.Column(scale=1):
                gr.HTML('<h4>⚙️ 固定提示词设置</h4>')
                fixed_prefix = gr.Textbox(
                    label='固定前缀提示词',
                    interactive=True,
                    placeholder='总是包含在开头的提示词...',
                    value=settings.DEFAULT_FIXED_PREFIX,
                    lines=2
                )
                fixed_suffix = gr.Textbox(
                    label='固定后缀提示词',
                    interactive=True,
                    placeholder='总是包含在结尾的提示词...',
                    value=settings.DEFAULT_FIXED_SUFFIX,
                    lines=2
                )
                enable_fixed = gr.Checkbox(
                    value=settings.DEFAULT_ENABLE_FIXED,
                    label='启用固定提示词',
                    interactive=True
                )

        # 高级设置手风琴
        with gr.Row():
            with gr.Accordion('🔧 高级设置', open=False):
                with gr.Column():
                    gr.HTML('<b>📊 生成设置</b>')
                    choices = wd_like.get_model_names()
                    if choices: wd_like.load_data(choices[-1])
                    dd_database = gr.Dropdown(
                        choices=choices,
                        value=choices[-1] if choices else None,
                        interactive=True,
                        label='数据库'
                    )
                    sl_tag_range = gr.Slider(
                        0, 8, 0, step=1, interactive=True,
                        label='标签数量过滤器'
                    )
                    txt_tag_range = gr.HTML(get_tag_range_txt(0))
                    nb_max_tag_num = gr.Number(
                        value=20, label='最大标签数',
                        precision=0, interactive=True
                    )

                with gr.Column():
                    gr.HTML('<b>🎛️ 调整参数</b>')
                    rb_prob_conversion_method = gr.Radio(
                        choices=['Cutoff and Power', 'Softmax'],
                        value='Cutoff and Power', type='index',
                        label='概率转换方法'
                    )
                    sl_power = gr.Slider(
                        0, 5, value=2, step=0.1,
                        label='权重强度', interactive=True
                    )
                    rb_sampling_method = gr.Radio(
                        choices=['NONE', 'Top-k', 'Top-p (Nucleus)'],
                        value='Top-k', type='index',
                        label='采样方法'
                    )
                    nb_k_value = gr.Number(
                        value=50, label='k值',
                        precision=0, interactive=True
                    )
                    sl_p_value = gr.Slider(
                        0, 1, label='p值',
                        value=0.1, step=0.01,
                        interactive=True
                    )
                    cb_weighted = gr.Checkbox(
                        value=True, label='使用权重选择',
                        interactive=True
                    )
                    cb_replace_underscore = gr.Checkbox(
                        value=True, label='用空格替换下划线',
                        interactive=True
                    )
                    cb_escape_brackets = gr.Checkbox(
                        value=True, label='转义括号',
                        interactive=True
                    )

    
        # 存储组件引用
        self.text2prompt_areas[i2i] = {
            'input': tb_input,
            'input_neg': tb_input_neg,
            'btn_generate': btn_generate,
            'btn_clear': btn_clear,
            'fixed_prefix': fixed_prefix,
            'fixed_suffix': fixed_suffix,
            'enable_fixed': enable_fixed,
            'database': dd_database,
            'tag_range': sl_tag_range,
            'tag_range_txt': txt_tag_range,
            'max_tags': nb_max_tag_num,
            'conversion': rb_prob_conversion_method,
            'power': sl_power,
            'sampling': rb_sampling_method,
            'k_value': nb_k_value,
            'p_value': sl_p_value,
            'weighted': cb_weighted,
            'replace_underscore': cb_replace_underscore,
            'escape_brackets': cb_escape_brackets
        }

        # 绑定事件
        dd_database.change(
            fn=dd_database_changed,
            inputs=[dd_database, sl_tag_range],
            outputs=[sl_tag_range, txt_tag_range]
        )

        sl_tag_range.change(
            fn=sl_tag_range_changed,
            inputs=sl_tag_range,
            outputs=txt_tag_range
        )

        nb_max_tag_num.change(
            fn=lambda x: max(0, x),
            inputs=nb_max_tag_num,
            outputs=nb_max_tag_num
        )

        nb_k_value.change(
            fn=lambda x: max(1, x),
            inputs=nb_k_value,
            outputs=nb_k_value
        )

        # 清空按钮事件
        btn_clear.click(
            fn=lambda: ("", "", "", ""),
            outputs=[tb_input, tb_input_neg, fixed_prefix, fixed_suffix]
        )

        # 生成按钮事件将在set_prompt_area中绑定，此时主提示词框引用还不可用

    def set_prompt_area(self, i2i: int, component: OnComponent):
        """保存主提示词框的引用并绑定事件"""
        self.prompt_area[i2i] = component.component
        print(f"[Text2Prompt Debug] 主提示词框引用已设置: i2i={i2i}")

        # 如果我们之前创建的UI区域已经存在，重新绑定生成按钮事件
        if self.text2prompt_areas[i2i] and self.text2prompt_areas[i2i]['btn_generate']:
            self._bind_generate_event(i2i)

    def _bind_generate_event(self, i2i: int):
        """绑定生成按钮事件到主提示词框"""
        try:
            area = self.text2prompt_areas[i2i]
            btn_generate = area['btn_generate']

            # 创建生成函数
            def generate_and_apply(*args):
                try:
                    result = self.prompt_gen_only(*args)
                    print(f"[Text2Prompt] 生成完成，长度: {len(result) if result else 0}")
                    return result
                except Exception as e:
                    print(f"[Text2Prompt] 错误: {str(e)}")
                    return f"错误: {str(e)}"

            # 重新绑定事件，输出到主提示词框
            btn_generate.click(
                fn=generate_and_apply,
                inputs=[
                    area['input'], area['input_neg'], area['fixed_prefix'],
                    area['fixed_suffix'], area['enable_fixed'], area['tag_range'],
                    area['conversion'], area['power'], area['sampling'],
                    area['max_tags'], area['k_value'], area['p_value'],
                    area['weighted'], area['replace_underscore'], area['escape_brackets']
                ],
                outputs=[self.prompt_area[i2i]]  # 直接输出到主提示词框
            )

            print(f"[Text2Prompt] 生成按钮事件已绑定到主提示词框")

        except Exception as e:
            print(f"[Text2Prompt] 绑定事件时出错: {str(e)}")

    
    def prompt_gen_only(self, *args):
        """生成提示词并组装最终结果"""
        # 解包参数
        (input_text, neg_text, fixed_prefix, fixed_suffix, enable_fixed,
         tag_range, conversion, power, sampling, max_tags, k_value, p_value,
         weighted, replace_underscore, escape_brackets) = args

        # 如果没有输入，返回空
        if not input_text.strip():
            return "请输入主题描述..."

        # 调用现有的生成逻辑
        generated_prompt = generate_prompt(
            input_text, neg_text, 1.0, tag_range, conversion, power,
            sampling, int(max_tags), int(k_value), p_value, weighted,
            replace_underscore, escape_brackets
        )

        # 组装最终提示词
        final_prompt = self.assemble_final_prompt(
            fixed_prefix, generated_prompt, fixed_suffix, enable_fixed
        )

        return final_prompt

    def assemble_final_prompt(self, prefix, generated, suffix, enable_fixed):
        """组装最终的提示词"""
        parts = []

        # 添加前缀
        if enable_fixed and prefix and prefix.strip():
            prefix_clean = prefix.strip()
            if not prefix_clean.endswith(','):
                prefix_clean += ','
            parts.append(prefix_clean)

        # 添加生成的内容
        if generated and generated.strip():
            parts.append(generated.strip())

        # 添加后缀
        if enable_fixed and suffix and suffix.strip():
            suffix_clean = suffix.strip()
            if not suffix_clean.startswith(',') and parts:
                suffix_clean = ',' + suffix_clean
            if not suffix_clean.endswith(','):
                suffix_clean += ','
            parts.append(suffix_clean)

        return ' '.join(parts)


# 注册脚本
def on_ui_tabs():
    # 返回空列表，因为我们使用Script方式而不是独立标签页
    return []

# 保持向后兼容性，但实际使用Script类
script_callbacks.on_ui_tabs(on_ui_tabs)