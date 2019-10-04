#打飛船射級遊戲
import pygame,random,math,time
#玩家操控的太空船
class Enemy(pygame.sprite.Sprite):
    speed=0
    def __init__(self,color,x,y,speed):
        pygame.sprite.Sprite.__init__(self)
        self.image=pygame.Surface([25,25])
        self.image.fill(color)
        self.rect=self.image.get_rect()
        self.rect.x=x
        self.rect.y=y
        self.speed=speed
    def update(self):
        self.rect.y+=self.speed
class Bullet(pygame.sprite.Sprite):
    def __init__(self,color,x,y,speed):
        pygame.sprite.Sprite.__init__(self)
        self.image=pygame.Surface([20,20])
        self.image.fill((255,255,255))
        pygame.draw.circle(self.image,(color),(10,10),5,0)
        self.rect=self.image.get_rect()
        self.rect.x=x
        self.rect.y=y
        self.speed=speed
    def update(self):
        self.rect.y-=self.speed
class Ship(pygame.sprite.Sprite):
    def __init__(self,color,x,y):
        pygame.sprite.Sprite.__init__(self)
        self.image=pygame.Surface([30,30])
        self.image.fill(color)
        self.rect=self.image.get_rect()#取得這個image的位置
        self.rect.x=x
        self.rect.y=y
    def update(self):
        #取得滑鼠的位置
        pos=pygame.mouse.get_pos()
        #把太空船的位置設成滑鼠左鍵按下的位置
        self.rect.x=pos[0]
        #不能越過邊界
        if self.rect.x>window.get_width()-self.rect.width:
            self.rect.x=window.get_width()-self.rect.width
        elif self.rect.x<0:
            self.rect.x=0
pygame.init()
font=pygame.font.SysFont("SimHei",100)
def gameover(message):
    global run
    text=font.render(message,1,(255,0,0))
    window.blit(text,(window.get_width()/2-150,window.get_height()/2))
    pygame.display.update()
    #run=False
    #time.sleep(3)
clock=pygame.time.Clock()#計時器
window=pygame.display.set_mode((1400,800))
pygame.display.set_caption("spacefight!")
background=pygame.Surface(window.get_size())#畫布
background=background.convert()#可有可無
background.fill((255,255,255))#畫布上色

window.blit(background,(0,0))#把畫布貼在繪圖視窗window上
pygame.display.update()


allsprite=pygame.sprite.Group()#角色群組變數
playersprite=pygame.sprite.Group()
enemysprite=pygame.sprite.Group()
bulletsprite=pygame.sprite.Group()
ship=Ship((255,0,0),window.get_width()/2,window.get_height()-30)
playersprite.add(ship)
allsprite.add(ship)
point=0#計分

playing=False#playing true代表球正在動
run=True#run false代表程式結束
n=0
while run:
    n+=1
    clock.tick(30)
    for event in pygame.event.get():
        if event.type==pygame.QUIT:#使用者者按x結束視窗
            run=False#跳出pygame
    button=pygame.mouse.get_pressed()
    #按下滑鼠左鍵開始遊戲（求開始動）
    if button[0]:
        playing=True
    if playing:
        window.blit(background,(0,0))#把background在貼到window上 等於清空視窗
        #button2=pygame.mouse.get_pressed()
        if button[2] and n>3:
            bullet=Bullet((255,0,0),ship.rect.x+5,ship.rect.y-10,10)
            bulletsprite.add(bullet)
            allsprite.add(bullet)
            n=0
        #隨機生成敵人
        for i in range(random.randint(0,2)):
            enemy=Enemy((random.randint(0,255),random.randint(0,255),random.randint(0,255)),random.randint(0,window.get_width()),0,random.randint(5,10))
            allsprite.add(enemy)
            enemysprite.add(enemy)
        #玩家碰到敵人就結束遊戲
        crash=pygame.sprite.spritecollide(ship,enemysprite,False)
        if len(crash)>0:
            break
        #把background在貼到window上 等於清空視窗
        window.blit(background,(0,0))
        #玩家移動
        ship.update()
        #子彈移動且探測有沒有撞到敵人 有撞到就兩個都不見且加分
        for i in bulletsprite:
            i.update()
            hitenemy=pygame.sprite.spritecollide(i,enemysprite,True)
            if len(hitenemy)>0:
                allsprite.remove(i)
                bulletsprite.remove(i)
                point+=len(hitenemy)
        #敵人往下跑
        for i in enemysprite:
            i.update()
            #敵人超出邊界刪除
            if i.rect.y>window.get_height():
                enemysprite.remove(i)
                allsprite.remove(i)
    allsprite.draw(window)
    msgscore="score: "+str(point)
    msgscoredisplay=font.render(msgscore,5,(255,0,0))
    window.blit(msgscoredisplay, (window.get_width()-350,0))
    pygame.display.update()
window.blit(background,(0,0))#把background在貼到window上 等於清空視窗
gameover('GGWP')
msgscore="score: "+str(point)
msgscoredisplay=font.render(msgscore,5,(255,0,0))
window.blit(msgscoredisplay, (window.get_width()-300,0))
allsprite.draw(window)
pygame.display.update()
time.sleep(4)
