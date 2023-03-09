import sys
from src.logger import logging

def error_message_detail(error,error_detail:sys):  ## error_detail will be present inside sys library
    _,_,exc_tb=error_detail.exc_info()  ##Talking about execution info and gives 3 important information and we are intrested in 3rd information
    file_name=exc_tb.tb_frame.f_code.co_filename ## This will fetch the required filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error)) ##exc_tb.tb_lineno --This will fetch the exact line number where exception has occured
    #str(error)) is the actual error which we provide here.

    return error_message

    

class CustomException(Exception): ##Inherting the parent Exception
    def __init__(self,error_message,error_detail:sys): ## Initialzing the constructor.
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail) ## error_message will be getting from our function
                                               ##error_detail will be tracked by sys library
    
    def __str__(self):
        return self.error_message ## For printing purpose
    