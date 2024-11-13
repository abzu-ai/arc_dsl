from arc_dsl.models.metrics import pixil_acc_bitmaps, pixil_acc_grids

def test_pixil_acc_grids():
    # Same grid
    acc = pixil_acc_grids("(make-grid 30 30 (make-bitmap 7 3 1 7 6 11 10))",
                           "(make-grid 30 30 (make-bitmap 7 3 1 7 6 11 10))")
    assert acc == 1.0

    # Different bitmap definitions, same grid rendered.
    acc = pixil_acc_grids("(make-grid 30 30 (make-bitmap 7 3 1 8 6 11 10 8))",
                           "(make-grid 30 30 (make-bitmap 7 3 1 7 6 11 10))")
    assert acc == 1.0

    # Different bitmaps, different renderings
    acc = pixil_acc_grids("(make-grid 30 30 (make-bitmap 1 2 1 7 6 11 10))",
                           "(make-grid 30 30 (make-bitmap 7 3 1 7 6 11 10))")
    assert acc < 1.0

    # Same bitmaps in differently sized grids
    acc = pixil_acc_grids("(make-grid 30 30 (make-bitmap 7 3 1 7 6 11 10))",
                           "(make-grid 30 29 (make-bitmap 7 3 1 7 6 11 10))")
    assert acc == (30 * 29) / (30*30)

def test_pixil_acc_bitmaps():
    # Same bitmaps
    acc = pixil_acc_bitmaps("(make-bitmap 7 3 1 7 6 11 10)",
                             "(make-bitmap 7 3 1 7 6 11 10)")
    assert acc == 1.0

    # Different bitmap definitions, same rendering.
    acc = pixil_acc_bitmaps("(make-bitmap 7 3 1 8 6 11 10 8)",
                             "(make-bitmap 7 3 1 7 6 11 10)")
    assert acc == 1.0

    # Different bitmaps
    acc = pixil_acc_bitmaps("(make-bitmap 1 2 1 7 6 11 10)",
                           "(make-bitmap 7 3 1 7 6 11 10)")
    assert acc < 1.0
