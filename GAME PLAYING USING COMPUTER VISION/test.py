from grabscreen import grab_screen
import cv2
def inverte(imagem, name):
    imagem = (255-imagem)
    cv2.imwrite(name, imagem)
while(True):
        # 800x600 windowed mode
        screen = grab_screen(region=(650,155,1250,257))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        screen = cv2.resize(screen, (60,60))
        screen = cv2.bitwise_not(screen)
        cv2.imshow('window2',screen)
        # resize to something a bit more acceptable for a CNN
        if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break