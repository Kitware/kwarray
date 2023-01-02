"""
"""
import kwplot
import kwimage
# import numpy as np
plt = kwplot.autoplt()
kwplot.autompl()

# lhs1 = kwimage.draw_text_on_image(None, 'k', color='kitware_blue')
lhs = kwimage.draw_text_on_image(None, 'kw', color='kitware_blue')
rhs1 = kwimage.draw_text_on_image(None, 'array', color='kitware_green')
rhs2 = kwimage.draw_text_on_image(None, 'array', color='kitware_green')
rhs = kwimage.stack_images([rhs1, rhs2], axis=0)

# poly1 = kwimage.Mask.coerce((lhs.sum(axis=2) > 0).astype(np.uint8)).to_multi_polygon()
# poly2 = kwimage.Mask.coerce((rhs.sum(axis=2) > 0).astype(np.uint8)).to_multi_polygon()
# poly1 = poly1.simplify(1)
# poly2 = poly2.simplify(1)

# poly2 = poly2.translate((0, poly1.to_box().br_y))
# box1 = poly1.to_box().to_polygon()
# box2 = poly2.to_box().to_polygon()

canvas = kwimage.stack_images([lhs, rhs], axis=1)

# box1.union(box2).to_box().scale(1.1, about='center').draw(fill=False, facecolor=None, setlim=1)

# poly1.draw(color='kitware_blue')
# poly2.draw(color='kitware_green')

# ax = plt.gca()
# ax.invert_yaxis()
# ax.set_aspect('equal')

# fig = plt.gcf()
# img = kwplot.render_figure_to_image(fig)
kwimage.imwrite('kwarray_logo.png', canvas)
kwplot.imshow(canvas)
