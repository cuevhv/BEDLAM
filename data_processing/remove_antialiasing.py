import OpenGL.GL
old_gl_enable = OpenGL.GL.glEnable
old_glRenderbufferStorageMultisample = OpenGL.GL.glRenderbufferStorageMultisample

suppress_multisampling = True
def new_gl_enable(value):
    if suppress_multisampling and value == OpenGL.GL.GL_MULTISAMPLE:
        OpenGL.GL.glDisable(value)
    else:
        old_gl_enable(value)

def new_glRenderbufferStorageMultisample(target, samples, internalformat, width, height):
    if suppress_multisampling:
        OpenGL.GL.glRenderbufferStorage(target, internalformat, width, height)
    else:
        old_glRenderbufferStorageMultisample(target, samples, internalformat, width, height)

OpenGL.GL.glEnable = new_gl_enable
OpenGL.GL.glRenderbufferStorageMultisample = new_glRenderbufferStorageMultisample
