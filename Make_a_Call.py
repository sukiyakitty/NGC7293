import sys
import serial
import serial.tools.list_ports
import time
import argparse


def open_SP_COM(spCOM='COM3'):
    finded = 0
    port_list = list(serial.tools.list_ports.comports())

    if len(port_list) <= 0:
        print('There are no COM Devices !!!')
        sys.exit(0)
    else:
        for port in port_list:
            port_inf = list(port)
            if (port_inf[0] == spCOM):
                finded = 1
                break

    if (finded == 1):
        print(spCOM)
        baudRate = 115200
        try:
            my_port = serial.Serial(spCOM, baudRate, timeout=1)
            print('NOW connecting ... > ', my_port.name, ' @ ', baudRate)
            if my_port.isOpen():
                print(my_port.name, " Open Success !")
                return my_port
            else:
                print(my_port.name, " Open Failed !")
        except:
            print(spCOM, 'was in USING !! Please close the last program.')
            sys.exit(0)
    else:
        print('NOT FOUND COM : ', spCOM)
        sys.exit(0)
    return


def open_fisrt():
    port_list = list(serial.tools.list_ports.comports())

    if len(port_list) <= 0:
        print('There are no COM Devices !!!')
        sys.exit(0)
    else:
        port_inf_0 = list(port_list[0])
        print(port_list[0])
        print(port_inf_0)
        port_serial_00 = port_inf_0[0]
        print(port_serial_00)
        baudRate = 115200
        try:
            my_port = serial.Serial(port_serial_00, baudRate, timeout=1)
            print('NOW connecting ... > ', my_port.name, ' @ ', baudRate)
            if my_port.isOpen():
                print(my_port.name, " Open Success !")
                return my_port
            else:
                print(my_port.name, " Open Failed !")

        except:
            print(port_serial_00, 'was in USING !! Please close the last program.')
            sys.exit(0)
    return


def open_myDevice(serialCOM_2='VID:PID=1A86:7523'):
    port_serial_myDevice = ''
    port_list = list(serial.tools.list_ports.comports())
    if len(port_list) <= 0:
        print('There are no COM Devices !!!')
        return None
    else:
        for port in port_list:
            port_inf = list(port)
            if (port_inf[2].find(serialCOM_2) != -1):
                port_serial_myDevice = port_inf[0]
    if (port_serial_myDevice != ''):
        print(port_serial_myDevice)
        baudRate = 115200
        try:
            my_port = serial.Serial(port_serial_myDevice, baudRate, timeout=1)
            print('NOW connecting ... > ', my_port.name, ' @ ', baudRate)
            if my_port.isOpen():
                print(my_port.name, " Open Success !")
                return my_port
            else:
                print(my_port.name, " Open Failed !")
                return None
        except:
            print(port_serial_myDevice, 'was in USING !! Please close the last program.')
            return None
    else:
        print('NOT FOUND Device : ', serialCOM_2)
        return None


def close_serial(my_port):
    my_port.close()
    if my_port.isOpen():
        print(my_port.name, " Close Failed !")
        return False
    else:
        print(my_port.name, " Close Success !")
        return True


def sendAT_Cmd(serInstance, atCmdStr, waitForRsp=0, maxTimeOut=0.1, timeSlice=0.1):
    print(atCmdStr)
    serInstance.write((atCmdStr + '\r').encode('utf-8'))
    if (waitForRsp == 1):
        result = waitForCmdRsp(serInstance, maxTimeOut, timeSlice)
    else:
        result = ''
    return result


def waitForCmdRsp(serInstance, maxTimeOut=0.1, timeSlice=0.1):
    rsp = ''
    time_remaining = maxTimeOut
    while (time_remaining > 0):
        time.sleep(timeSlice)
        time_remaining = time_remaining - timeSlice
        rsp += serInstance.read_all().decode()
        if not rsp == '':
            break
    print(rsp)
    return rsp


def main(args):
    phoneNum = args.phoneNum
    msg = 'AT+CTTS=1,"' + args.content + '"'
    try_times = 3
    failure_wait_seconds = 30
    msg_lasting = len(args.content) * 0.075 + 6
    my_USB_GSM = open_myDevice()
    if my_USB_GSM is not None:
        sendAT_Cmd(my_USB_GSM, 'AT', 1, 0.1, 0.1)
        sendAT_Cmd(my_USB_GSM, 'AT', 1, 0.1, 0.1)
        sendAT_Cmd(my_USB_GSM, 'AT', 1, 0.1, 0.1)
        if (sendAT_Cmd(my_USB_GSM, 'AT', 1, 0.1, 0.1).find('OK') != -1):
            print('CONNECTED!!!')
        sendMsgSuccess = 0
        for i in range(try_times):
            sendAT_Cmd(my_USB_GSM, 'ATD' + phoneNum + ';', 1, 0.1, 0.1)
            if (waitForCmdRsp(my_USB_GSM, 90, 0.5).find('+COLP: ') != -1):  # find '+COLP: '
                if (sendAT_Cmd(my_USB_GSM, msg, 1, 0.5, 0.5).find('OK') != -1):
                    replyMessage = waitForCmdRsp(my_USB_GSM, msg_lasting, 8)
                    if (replyMessage.find('NO CARRIER') != -1):
                        sendAT_Cmd(my_USB_GSM, 'ATH', 1, 0.1, 0.1)
                        print('Dialing Faled! time:', i, '!!!')
                        print('Retry!!!')
                        time.sleep(failure_wait_seconds)
                        continue
                    if (replyMessage.find('+CTTS: 0') != -1):
                        sendAT_Cmd(my_USB_GSM, 'ATH', 1, 0.1, 0.1)
                        sendMsgSuccess = 1
                        break
                    else:
                        sendAT_Cmd(my_USB_GSM, 'ATH', 1, 0.1, 0.1)
                        print('Dialing Faled! time:', i, '!!!')
                        print('Retry!!!')
                        time.sleep(failure_wait_seconds)
                        continue
                else:
                    sendAT_Cmd(my_USB_GSM, 'ATH', 1, 0.1, 0.1)
                    print('Dialing Faled! time:', i, '!!!')
                    print('Retry!!!')
                    time.sleep(failure_wait_seconds)
                    continue
            else:  # not find '+COLP: '
                sendAT_Cmd(my_USB_GSM, 'ATH', 1, 0.1, 0.1)
                print('Dialing Failed! time:', i, '!!!')
                print('Retry!!!')
                time.sleep(failure_wait_seconds)
                continue
        if (sendMsgSuccess == 1):
            print('Success called !!!')
        else:
            print('Failed called !!!')
        close_serial(my_USB_GSM)


def parseArguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--phoneNum', type=str, nargs='?', default='17710805067',
                        help='The telephone number you want to call.')
    parser.add_argument('--content', type=str, nargs='?',
                        default='4F60597DFF0C6211662F004300440037FF0C8B66544AFF0C4E8C6C27531678B36D535EA64F4E4E8E767E52064E4B0032002E0035FF0C8BF7901F676559047406FF0C8B66544AFF0C4E8C6C27531678B36D535EA64F4E4E8E767E52064E4B0032002E0035FF0C8BF7901F676559047406',
                        help='The message you want to deliver.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parseArguments(sys.argv[1:]))
