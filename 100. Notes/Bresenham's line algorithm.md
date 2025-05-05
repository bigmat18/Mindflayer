**Data time:** 16:02 - 22-10-2024

**Status**: #note #youngling 

**Tags:** [[Rendering Lines]]

**Area**: [[Bachelor's Degree]]

# Bresenham's line algorithm

The Bresenham's line algorithm is a line drawing algorithm commonly used to draw line primitives in a bitmap image. It's vary simple and efficient but have some limitation because depend on hardware implementation. An extensions is [[Xiaolin Wu's line algorithm]] that provide the antialias.

## Math basics
We use the following conventions:
- **top-left** is (0,0) and pixel value increate in the righe and down directions
- the pixel **center have integer cordinates**
- We call the two endpoints of the line $(x_{0}, y_{0})$ and $(x_1, y_1)$ 

The slop-intercept form of a line is written as:
$$y = f(x) = mx + b$$
We need an equation that take (x,y) value to able to draw a line at any algle. The angle (or slope) can be written like $\frac{\Delta y}{\Delta x}$. The using algebric manipulation:
$$y = \frac{\Delta y}{\Delta x}x + b \:\:\: \Rightarrow \:\:\: (\Delta x)y = (\Delta y)x + (\Delta x)b \:\:\:\Rightarrow \:\:\: 0 = (\Delta y)x - (\Delta x)y + (\Delta x)b$$
This equation can be a function of x and y, it can be written as:
$$f(x,y) = Ax + By + C$$
- $A = \Delta y = y_{1} - y_{0}$
- $B = -\Delta x = -(x_{1} - x_{0})$
- $C = (\Delta x)b = (x_{1} - x_{0})b$

With this equation if we have $f(x_{0},y_{0}) = 0$ the point $(x, y)$ is on the line otherwise not. 
## Algorithm
We have that the starting points it on the line ($f(x,y) = 0$). From this we have two option to choose:
- point in position $(x_0 + 1, y_0)$ 
- point in position $(x_0+1, y_0+1)$ 

![[Screenshot 2024-10-22 at 16.45.30.png | 250]]

The point should be chosen based upon which is closer to the line at $x_0 + 1$. To resolve this we can evaluate the line function at the midpoint between these two points $f(x_0+1, y_0 + \frac{1}{2})$.

- If the value is positive the ideal line is below the midpoints and closer to  $(x_0+1, y_0+1)$ 
- otherwise the ideal line passes though or abode the midpoints ant the y coordinate should stay the same, in which case we choose the point $(x_0 + 1, y_0)$ 

## Algorithm for integer arithmetic
In alternative we cane calculate the difference between points. This method is generally faster using integer-onl arithmetic instead using floating-point arithmetic. We define
$$D_{i} = f( x_{i} + 1, y_{i} + \frac{1}{2}) - f(x_{0}, y_{0})$$
- if $(x_0 + 1, y_0)$ is the chosen the change in $D_i$ will be:
$$\Delta D = f\left( x_{0} + 2, y_{0} + \frac{1}{2} \right) - f\left( x_{0} + 1, y_{0} + \frac{1}{2} \right) = A = \Delta y$$
- if $(x_0 + 1, y_0 + 1)$ is the chosen the change in $D_i$ will be:
$$\Delta D = f\left( x_{0} + 2, y_{0} + \frac{3}{2} \right) - f\left( x_{0} + 1, y_{0} + \frac{1}{2} \right) = A + B = \Delta y - \Delta y$$

This results in an algorithm that uses only integer arithmetic.
```
plotLine(x0, y0, x1, y1)
    dx = x1 - x0
    dy = y1 - y0
    D = 2*dy - dx // from manipulation we can found this fomula for first point
    y = y0

    for x from x0 to x1
        plot(x, y)
        if D > 0
            y = y + 1
            D = D - 2*dx
        end if
        D = D + 2*dy
```

For a general situation that work with line not only with octant zero we have:
```
plotLine(x0, y0, x1, y1)
    if abs(y1 - y0) < abs(x1 - x0)
        if x0 > x1
            plotLineLow(x1, y1, x0, y0)
        else
            plotLineLow(x0, y0, x1, y1)
        end if
    else
        if y0 > y1
            plotLineHigh(x1, y1, x0, y0)
        else
            plotLineHigh(x0, y0, x1, y1)
        end if
    end if


plotLineHigh(x0, y0, x1, y1)
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0
        xi = -1
        dx = -dx
    end if
    D = (2 * dx) - dy
    x = x0

    for y from y0 to y1
        plot(x, y)
        if D > 0
            x = x + xi
            D = D + (2 * (dx - dy))
        else
            D = D + 2*dx
        end if


plotLineLow(x0, y0, x1, y1)
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0
        yi = -1
        dy = -dy
    end if
    D = (2 * dy) - dx
    y = y0

    for x from x0 to x1
        plot(x, y)
        if D > 0
            y = y + yi
            D = D + (2 * (dy - dx))
        else
            D = D + 2*dy
        end if
```
# References
- [Wikipedia page](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm)