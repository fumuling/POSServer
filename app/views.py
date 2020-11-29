from django.shortcuts import render

# Create your views here.

from POS.POS_func import get_list, pos_text_seg, pos_text_predict

def pos_list(request):
    text = request.POST.get('text')

    if text:
        segged_text = pos_text_seg(text)
        final_output = pos_text_predict(segged_text)
        word_list, POS_list = get_list(final_output)
    else:
        word_list = ["你", '的', '妈妈']
        POS_list = ['NN', 'AS', 'SD']
    res_list = {
        'word_list': word_list,
        'POS_list': POS_list
    }
    return render(request, '前端.html', res_list)





