import sys
import time
from math import cos, sin

if __name__ == "__main__":
    try:
        import turtle as trtl
    except Exception:
        print("turtle graphics not found or some other related error, skipping")
        sys.exit(0)

    class Star(trtl.Turtle):
        def __init__(self, size=6):
            super().__init__(shape="circle")
            self.angle = size
            self.orbitRadius = size
            self.c = "#f100f1"
            self.color(self.c)
            self.size = size
            self.shapesize(size, size)
            self.name = "ERRORNAME"

    class Planet(trtl.Turtle):
        def __init__(self, size, orbitRadius, primary):
            super().__init__(shape="circle")
            self.angle = size
            self.orbitRadius = orbitRadius
            self.c = "#ffffff"
            self.color(self.c)
            self.size = size
            self.shapesize(size, size)
            self.primary = primary
            self.name = "ERRORNAME"
            self.orbitalSpeed = self.orbitRadius
            self.orbitalSpeed = ((4 / self.orbitRadius) ** 0.5) * (20 / 400)

        def move(self):
            x = self.orbitRadius * cos(self.angle)  # Angle in radians
            y = self.orbitRadius * sin(self.angle)
            self.goto(self.primary.xcor() + x, self.primary.ycor() + y)

    def main():
        SCREEN_X = 1200
        SCREEN_Y = 800

        star = Star()
        screen = trtl.Screen()
        screen.setup(SCREEN_X, SCREEN_Y, startx=50, starty=50)
        screen.bgcolor("black")
        screen.tracer(0)

        displayListplan = [
            Planet(1, 222, star),
            Planet(2, 255, star),
            Planet(3, 333, star),
        ]
        for y in displayListplan:
            y.orbitalSpeed = ((4 / y.orbitRadius) ** 0.5) * (20 / 400)
            y.penup()
            y.move()
            y.pendown()

        while True:
            try:
                screen.update()
            except Exception:
                ...

            for i in displayListplan:
                i.move()
            for p in displayListplan:
                p.angle += p.orbitalSpeed

            # for i in displayListplan:
            #     i.move()
            # for i in displayListstar:
            #     i.move()
            # for p in displayListplan:
            #     p.angle += p.orbitalSpeed
            # if bianaryCheck == 0:
            star.angle += 1  # coorbitSpeed
            # displayStar2.angle += coorbitSpeed
            # star2nameTag.move()
            # star1nameTag.move()
            # starAttributesTag.displayBianary()
            # else:
            #     pass
            # screen.onkey(starAttributesTag.toggle, "s")
            # # screen.onkey(starAttributesTag.displaySingle, "s")
            # if len(displayListplan) > 0:
            #     screen.onkey(planetCursor.moveup, "space")
            #     screen.onkey(planetCursor.select, "Return")
            # else:
            #     pass
            # try:
            #     screen.listen()
            # except:
            #     ...

            time.sleep(1 / 30)

    main()
